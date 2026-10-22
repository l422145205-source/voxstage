"""Real local inference; reference-free ASR; probe/group split frozen before first inference."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'runtime/deps/common')]
from voxstage.audio import digest
from voxstage.contracts import Scene
from voxstage.director import Director
from voxstage.store import atomic_json
from voxstage.workers import ModelBackend


def main():
    probes=ROOT/'examples/probes.json';data=json.loads(probes.read_text(encoding='utf-8'))
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    freeze={'dataset_sha256':digest(probes),'director_sha256':digest(ROOT/'src/voxstage/director.py'),
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'policy':data['policy'],
        'split':'space and forest for development; island and clockwork for holdout; no human listening labels'}
    if (reports/'protocol_freeze.json').exists():
        old=json.loads((reports/'protocol_freeze.json').read_text())
        assert old['dataset_sha256']==freeze['dataset_sha256'],'frozen probes changed'
    else:atomic_json(reports/'protocol_freeze.json',freeze)
    backend=ModelBackend(ROOT);records=[];showcases=[]
    try:
        for index,case in enumerate(data['cases']):
            scene=Scene.model_validate({'id':case['id'],'title':case['id'],'cast':[
                {'id':'actor','name':'Original probe','speaker':case['speaker']}],
                'cues':[{'id':'line','character':'actor','text':case['text'],'language':case['language'],
                         'start':0,'end':case['capacity'],'tone':case['tone']}],
                'policy':data['policy'],'seed':3000+index})
            result=Director(ROOT,backend).run(scene)
            records.append({'case':case,'result':result})
            atomic_json(reports/'pilot-progress.json',records)
            print(case['id'],result['status'],'takes',len(result['takes']),flush=True)
        for name in ['spaceport','clockwork']:
            scene=Scene.model_validate_json((ROOT/'examples'/f'{name}.json').read_text(encoding='utf-8'))
            director=Director(ROOT,backend);result=director.run(scene)
            print('showcase',name,result['status'],flush=True)
            replay=Director(ROOT,backend).run(scene)
            edited=scene.model_copy(deep=True)
            if name=='spaceport':
                # A real one-line edit; untouched cues should remain cache hits.
                edited.cues[0].text='指挥官，所有乘客已经登船。'
                edit_result=Director(ROOT,backend).run(edited)
            else:edit_result=None
            showcases.append({'scene':scene.model_dump(mode='json'),'result':result,
                'replay_stats':replay['stats'],'one_line_edit':edit_result})
        summary={}
        for split in ['development','holdout']:
            subset=[r for r in records if r['case']['split']==split]
            first=[r['result']['takes'][0] for r in subset]
            baseline=sum(not t['issues'] and t['raw_signal']['duration']<=r['case']['capacity']
                         for r,t in zip(subset,first))
            first_fit=sum(not t['issues'] for t in first)
            adaptive=sum(r['result']['status']=='complete' for r in subset)
            summary[split]={'n':len(subset),'one_shot_no_timing_fit_passed':baseline,
                'one_shot_with_bounded_fit_passed':first_fit,'bounded_repair_passed':adaptive,
                'mean_first_take_asr_cer':statistics.mean(t['content']['cer'] for t in first),
                'tts_calls':sum(r['result']['stats']['tts_calls'] for r in subset),
                'asr_calls':sum(r['result']['stats']['asr_calls'] for r in subset)}
        report={'protocol':freeze,'backend_identity':backend.identity,
            'worker_info':{'tts':backend.tts.info,'asr':backend.asr.info},
            'summary':summary,'records':records,'showcases':showcases,
            'interpretation':'ASR-based content and deterministic timing/acoustic checks only. No human MOS, emotion accuracy, speaker similarity, fine-tuning or production traffic claims.'}
        atomic_json(reports/'production_pilot.json',report)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    finally:backend.close()


if __name__=='__main__':main()
