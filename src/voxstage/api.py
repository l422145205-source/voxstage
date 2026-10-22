from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .contracts import Scene
from .director import Director


def create_app(root,backend=None):
    root=Path(root).resolve();pool=ThreadPoolExecutor(max_workers=1)
    jobs={};lock=Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True)
        if backend:backend.close()

    app=FastAPI(title='VoxStage',version='0.1.0',lifespan=lifespan)

    @app.get('/api/config')
    def config():
        return {'models_available':bool(backend),'default_mode':'local_models' if backend else 'saved_results',
                'scope':'loopback single-user dialogue production prototype'}

    @app.get('/api/scenes')
    def scenes():
        result=[]
        for path in sorted((root/'examples').glob('*.json')):
            try:
                scene=Scene.model_validate_json(path.read_text(encoding='utf-8'))
                result.append(scene.model_dump(mode='json'))
            except (ValueError,KeyError):pass
        return result

    @app.get('/api/runs')
    def runs():
        results=[]
        for folder in [root/'data/runs',root/'examples/saved']:
            for path in sorted(folder.glob('*/result.json')):
                value=json.loads(path.read_text(encoding='utf-8'))
                results.append({'run_id':value['run_id'],'scene_id':value['scene_id'],
                    'status':value['status'],'backend':value['backend']})
        return results[-40:]

    def find_run(run_id):
        import re
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}',run_id):
            raise HTTPException(404,'unknown run')
        for prefix in [root/'data/runs',root/'examples/saved']:
            folder=prefix/run_id
            if (folder/'result.json').is_file():return folder
        raise HTTPException(404,'unknown run')

    @app.get('/api/runs/{run_id}')
    def run(run_id:str):
        folder=find_run(run_id)
        return {'result':json.loads((folder/'result.json').read_text(encoding='utf-8')),
                'scene':json.loads((folder/'scene.json').read_text(encoding='utf-8'))}

    @app.get('/api/runs/{run_id}/export/{filename}')
    def export(run_id:str,filename:str):
        if filename not in {'dialogue.wav','dialogue.srt','timeline.json'}:
            raise HTTPException(404,'unknown export')
        folder=find_run(run_id)
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        if result['status']!='complete':raise HTTPException(409,'scene has unresolved cues')
        path=folder/'export'/filename
        if not path.is_file():raise HTTPException(409,'scene has unresolved cues')
        if filename=='dialogue.wav':
            from .audio import digest
            if digest(path)!=result['export']['audio']['sha256']:
                raise HTTPException(409,'published audio hash no longer matches its record')
        return FileResponse(path,filename=filename)

    @app.get('/api/runs/{run_id}/takes/{index}/{variant}')
    def take_audio(run_id:str,index:int,variant:str):
        folder=find_run(run_id);result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        if not 0<=index<len(result['takes']) or variant not in {'raw','fitted'}:
            raise HTTPException(404,'unknown take')
        take=result['takes'][index]
        if (folder/'takes').is_dir():
            path=folder/'takes'/f'{index}-{variant}.wav'
        else:
            if variant=='raw':path=root/'data/cache/tts'/take['take_key']/'raw.wav'
            else:path=Path(take.get('path','')).resolve()
            if not path.is_relative_to(root/'data/cache/tts'):
                raise HTTPException(404,'unknown take path')
        if not path.is_file():raise HTTPException(404,'take unavailable')
        return FileResponse(path,media_type='audio/wav')

    @app.post('/api/jobs',status_code=202)
    def create_job(scene:Scene):
        if backend is None:raise HTTPException(409,'models are not configured; open a saved run')
        with lock:
            if any(j['status'] in {'queued','running'} for j in jobs.values()):
                raise HTTPException(409,'one GPU job is already active')
            job_id=uuid.uuid4().hex;jobs[job_id]={'status':'queued','events':[]}
        def work():
            with lock:jobs[job_id]['status']='running'
            def event(value):
                with lock:
                    if len(jobs[job_id]['events'])<500:jobs[job_id]['events'].append(value)
            try:
                result=Director(root,backend,event).run(scene)
                with lock:jobs[job_id].update(status='complete',run_id=result['run_id'])
            except Exception as exc:
                with lock:jobs[job_id].update(status='failed',error=str(exc)[:1500])
        pool.submit(work)
        return {'job_id':job_id,'status':'queued'}

    @app.get('/api/jobs/{job_id}')
    def job(job_id:str):
        with lock:
            if job_id not in jobs:raise HTTPException(404,'unknown job')
            return dict(jobs[job_id])

    app.mount('/',StaticFiles(directory=Path(__file__).parent/'static',html=True),name='ui')
    return app
