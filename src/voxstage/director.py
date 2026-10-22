"""Bounded production policy: observe -> classify -> repair, without rewriting dialogue."""
from pathlib import Path
import json
import time

from . import audio
from .contracts import INSTRUCTIONS, Repair, Scene
from .store import atomic_json, exclusive, key
from .text import compare,normalize


def diagnose(cue,policy,signal,content):
    reasons=[]
    if signal['sample_rate']!=audio.RATE:reasons.append('sample_rate')
    if signal['rms_dbfs']<policy.min_rms_dbfs:reasons.append('silence_or_low_energy')
    if signal['clipped_fraction']>policy.max_clipped_fraction:reasons.append('clipping')
    if content['cer']>policy.max_cer:reasons.append('content_mismatch')
    missing=[t for t in cue.protected_terms if normalize(t) not in content['normalized_hypothesis']]
    content['missing_protected_terms']=missing
    if missing:reasons.append('protected_term_missing')
    if signal['duration']>cue.capacity*policy.max_speedup:reasons.append('duration_overflow')
    return reasons


def decide(cue,issues,attempt,budget):
    if not issues:return Repair(cue_id=cue.id,action='accept',reason='objective constraints passed')
    if attempt>=budget:return Repair(cue_id=cue.id,action='reject',reason=', '.join(issues))
    action='faster' if set(issues)<={'duration_overflow','transform_not_feasible','final_duration_overflow'} else 'resynthesize'
    return Repair(cue_id=cue.id,action=action,reason=', '.join(issues))


class Director:
    def __init__(self,root,backend,on_event=None):
        self.root=Path(root).resolve();self.backend=backend;self.on_event=on_event or (lambda event:None)
        self.cache=self.root/'data/cache';self.cache.mkdir(parents=True,exist_ok=True)
        self.stats={'tts_calls':0,'asr_calls':0,'tts_cache_hits':0,'asr_cache_hits':0}

    def event(self,name,**values):
        self.on_event({'event':name,**values})

    def transcribe(self,path,language):
        fingerprint=key({'version':1,'backend':self.backend.identity,'audio_sha256':audio.digest(path),
                         'language':language,'context':'empty'})
        target=self.cache/'asr'/fingerprint/'result.json'
        with exclusive(target.with_suffix('.lock')):
            if target.exists():
                result=json.loads(target.read_text(encoding='utf-8'))
                if result.get('request_fingerprint')==fingerprint and result.get('reference_supplied') is False:
                    self.stats['asr_cache_hits']+=1;return result
            self.stats['asr_calls']+=1
            result=self.backend.transcribe(audio=path,language=language)
            if result.get('reference_supplied') is not False:
                raise ValueError('ASR verification must not receive the reference')
            result['request_fingerprint']=fingerprint
            atomic_json(target,result)
            return result

    def take(self,scene,cue,speaker,attempt,previous_issues):
        instruction=INSTRUCTIONS[cue.tone]
        if attempt>1:
            if previous_issues==['duration_overflow']:
                instruction+=' 语速快一些，保持吐字清楚。Speak faster while remaining clear.'
            else:
                instruction+=' 逐字清晰朗读，不漏字、不重复。Articulate every word clearly without omissions or repetitions.'
        seed=(scene.seed+attempt*1009+int(key(cue.id)[:6],16))%(2**31-1)
        parameters={'text':cue.spoken,'language':cue.language,'speaker':speaker,'instruct':instruction,'seed':seed}
        take_key=key({'version':1,'backend':self.backend.identity,**parameters})
        folder=self.cache/'tts'/take_key; raw=folder/'raw.wav';meta=folder/'take.json'
        folder.mkdir(parents=True,exist_ok=True)
        with exclusive(folder/'take.lock'):
            cached=None
            if meta.exists() and raw.exists():
                candidate=json.loads(meta.read_text(encoding='utf-8'))
                if candidate.get('sha256')==audio.digest(raw):cached=candidate
            if cached is None:
                self.event('synthesize',cue_id=cue.id,attempt=attempt)
                self.stats['tts_calls']+=1
                result=self.backend.synthesize(**parameters,output=raw)
                signal=audio.analyze(raw)
                cached={'take_key':take_key,'sha256':signal['sha256'],'parameters':parameters,'synthesis':result}
                atomic_json(meta,cached)
            else:
                self.stats['tts_cache_hits']+=1;self.event('reuse_take',cue_id=cue.id,attempt=attempt)
        self.event('verify_content',cue_id=cue.id,attempt=attempt)
        signal=audio.analyze(raw)
        transcript=self.transcribe(raw,cue.language)
        content=compare(cue.spoken,transcript['text'])
        issues=diagnose(cue,scene.policy,signal,content)
        result={'take_key':take_key,'cue_id':cue.id,'attempt':attempt,'raw_signal':signal,
            'asr':transcript,'content':content,'issues':issues,'parameters':parameters}
        if not issues:
            fit_key=key({'version':1,'take_sha256':signal['sha256'],'capacity':cue.capacity,
                         'max_speedup':scene.policy.max_speedup})
            fitted=folder/(fit_key+'.wav')
            try:
                with exclusive(folder/(fit_key+'.lock')):
                    transform=audio.speed_fit(raw,fitted,cue.capacity,scene.policy.max_speedup)
                    final_signal=audio.analyze(fitted)
                    final_asr=self.transcribe(fitted,cue.language)
                    final_content=compare(cue.spoken,final_asr['text'])
                    issues=diagnose(cue,scene.policy,final_signal,final_content)
                    if final_signal['duration']>cue.capacity+1/audio.RATE:issues.append('final_duration_overflow')
            except ValueError as exc:
                result.update(issues=['transform_not_feasible'],transform_error=str(exc))
                return result
            result.update(transform=transform,final_signal=final_signal,final_asr=final_asr,
                          final_content=final_content,issues=issues,path=str(fitted),sha256=final_signal['sha256'])
        return result

    def run(self,scene,*,interrupt_after_cue=None):
        scene=Scene.model_validate(scene.model_dump(mode='json') if isinstance(scene,Scene) else scene)
        fingerprint=key({'version':1,'scene':scene.model_dump(mode='json'),'backend':self.backend.identity})
        folder=self.root/'data/runs'/(scene.id+'-'+fingerprint[:16])
        folder.mkdir(parents=True,exist_ok=True)
        with exclusive(folder/'scene.lock'):
            return self._run(scene,folder,fingerprint,interrupt_after_cue)

    def _run(self,scene,folder,fingerprint,interrupt_after_cue):
        started=time.perf_counter();self.stats={k:0 for k in self.stats}
        atomic_json(folder/'scene.json',scene.model_dump(mode='json'))
        cast={c.id:c.speaker for c in scene.cast};accepted={};records=[];rejected=[]
        for cue in scene.cues:
            issues=[]
            for attempt in range(1,scene.policy.max_attempts+1):
                result=self.take(scene,cue,cast[cue.character],attempt,issues)
                issues=result['issues'];decision=decide(cue,issues,attempt,scene.policy.max_attempts)
                result['decision']=decision.model_dump();records.append(result)
                atomic_json(folder/'takes.json',records)
                self.event('decision',cue_id=cue.id,attempt=attempt,**decision.model_dump(exclude={'cue_id'}))
                if decision.action=='accept':accepted[cue.id]=result;break
                if decision.action=='reject':rejected.append(cue.id);break
            if interrupt_after_cue==cue.id:
                raise RuntimeError('controlled interruption after cue '+cue.id)
        result={'scene_id':scene.id,'run_id':folder.name,'request_fingerprint':fingerprint,
            'backend':self.backend.label,'backend_identity':self.backend.identity,
            'status':'needs_review' if rejected else 'complete','rejected_cues':rejected,
            'accepted_cues':list(accepted),'stats':self.stats.copy(),
            'elapsed_seconds':time.perf_counter()-started,
            'subjective_quality':'style, acting and naturalness require human listening',
            'takes':records}
        if not rejected:
            result['export']=audio.export_scene(scene,accepted,folder/'export')
        atomic_json(folder/'result.json',result)
        self.event('finished',run_id=folder.name,status=result['status'])
        return result
