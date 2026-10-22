"""Recheck frozen predictions against final contracts; package real replay assets."""
from pathlib import Path
import copy
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'runtime/deps/common')]
from voxstage.contracts import Scene
from voxstage.director import Director
from voxstage.audio import digest
from voxstage.store import atomic_json
from voxstage.workers import ModelBackend


def portable(value):
    text=json.dumps(value,ensure_ascii=False)
    text=text.replace(str(ROOT).replace('\\','\\\\'),'${PROJECT_ROOT}')
    return json.loads(text)


def save_run(scene,result):
    folder=ROOT/'examples/saved'/result['run_id'];folder.mkdir(parents=True,exist_ok=True)
    for i,take in enumerate(result['takes']):
        raw=ROOT/'data/cache/tts'/take['take_key']/'raw.wav'
        (folder/'takes').mkdir(exist_ok=True)
        shutil.copyfile(raw,folder/'takes'/f'{i}-raw.wav')
        if take.get('path'):shutil.copyfile(take['path'],folder/'takes'/f'{i}-fitted.wav')
    if result['status']=='complete':
        source=ROOT/'data/runs'/result['run_id']/'export'
        (folder/'export').mkdir(exist_ok=True)
        for name in ['dialogue.wav','dialogue.srt','timeline.json']:
            shutil.copyfile(source/name,folder/'export'/name)
    atomic_json(folder/'scene.json',scene)
    atomic_json(folder/'result.json',portable(result))


def main():
    initial=json.loads((ROOT/'reports/production_pilot.json').read_text(encoding='utf-8'))
    backend=ModelBackend(ROOT);replay=[]
    try:
        for record in initial['records']:
            case=record['case'];original=record['result']
            scene=Scene.model_validate_json((ROOT/'data/runs'/original['run_id']/'scene.json').read_text(encoding='utf-8'))
            result=Director(ROOT,backend).run(scene)
            assert result['status']=='complete',case['id']
            assert result['stats']['tts_calls']==0 and result['stats']['asr_calls']==0
            replay.append({'case':case['id'],'run_id':result['run_id'],'status':result['status'],'stats':result['stats']})
        case=next(r for r in initial['records'] if r['case']['id']=='test07')
        stress=Scene.model_validate_json((ROOT/'data/runs'/case['result']['run_id']/'scene.json').read_text(encoding='utf-8'))
        stress.id='deadline';stress.title='时间槽不可满足 · 真实候选回放';stress.cues[0].end=.5
        result=Director(ROOT,backend).run(stress)
        assert result['status']=='needs_review' and 'export' not in result
        atomic_json(ROOT/'examples/deadline.json',stress.model_dump(mode='json'))
        save_run(stress.model_dump(mode='json'),result)
        for showcase in initial['showcases']:
            save_run(showcase['scene'],showcase['result'])
        atomic_json(ROOT/'reports/final_constraints_replay.json',{
            'description':'Re-evaluated original real predictions after contract hardening; no new model inference',
            'director_sha256':digest(ROOT/'src/voxstage/director.py'),'records':replay,
            'stress':{'status':result['status'],'stats':result['stats'],'source_probe':'test07',
                'purpose':'Deliberately impossible deadline using two existing real takes; not a new quality sample'}})
        provenance={kind:json.loads((ROOT/'runtime/models'/kind/'download_manifest.json').read_text()) for kind in ['tts','asr']}
        atomic_json(ROOT/'reports/model_provenance.json',provenance)
        print('packaged real saved runs',len(initial['showcases'])+1,flush=True)
    finally:backend.close()


if __name__=='__main__':main()
