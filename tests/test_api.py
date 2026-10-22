import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
from voxstage.api import create_app

def test_saved_mode_and_unknown_paths(tmp_path):
    (tmp_path/'examples').mkdir()
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/config').json()['models_available'] is False
        assert client.get('/api/runs/missing').status_code==404
        assert client.get('/api/runs/missing/export/private.txt').status_code==404
        assert client.get('/').status_code==200

def test_real_media_api_contract_with_explicit_fixture(tmp_path,scene,backend):
    (tmp_path/'examples').mkdir()
    (tmp_path/'examples/test.json').write_text(scene.model_dump_json(),encoding='utf-8')
    with TestClient(create_app(tmp_path,backend)) as client:
        assert len(client.get('/api/scenes').json())==1
        response=client.post('/api/jobs',json=scene.model_dump(mode='json'))
        assert response.status_code==202
        job=response.json()['job_id']
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            state=client.get('/api/jobs/'+job).json()
            if state['status'] in ['complete','failed']:break
            time.sleep(.02)
        assert state['status']=='complete',state
        run=state['run_id']
        assert client.get(f'/api/runs/{run}').json()['result']['status']=='complete'
        assert client.get(f'/api/runs/{run}/export/dialogue.wav',headers={'Range':'bytes=0-63'}).status_code==206
        assert client.get(f'/api/runs/{run}/takes/0/raw').status_code==200
        assert client.get(f'/api/runs/{run}/takes/999/raw').status_code==404
        assert client.get(f'/api/runs/{run}/takes/0/secret').status_code==404
        assert client.get(f'/api/runs/{run}/export/dialogue.srt').status_code==200

def test_no_model_fallback_and_invalid_scene(tmp_path,scene):
    (tmp_path/'examples').mkdir()
    with TestClient(create_app(tmp_path)) as client:
        assert client.post('/api/jobs',json=scene.model_dump(mode='json')).status_code==409
        value=scene.model_dump(mode='json');value['shell_command']='echo unsafe'
        assert client.post('/api/jobs',json=value).status_code==422

def test_rejected_result_never_serves_stale_export(tmp_path,scene,backend):
    from voxstage.director import Director
    result=Director(tmp_path,backend).run(scene)
    folder=tmp_path/'data/runs'/result['run_id']
    result['status']='needs_review'
    (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f"/api/runs/{result['run_id']}/export/dialogue.wav").status_code==409

def test_published_audio_integrity_is_enforced(tmp_path,scene,backend):
    from voxstage.director import Director
    result=Director(tmp_path,backend).run(scene)
    folder=tmp_path/'data/runs'/result['run_id']
    (folder/'export/dialogue.wav').write_bytes(b'changed')
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f"/api/runs/{result['run_id']}/export/dialogue.wav").status_code==409
