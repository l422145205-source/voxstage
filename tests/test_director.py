from pathlib import Path
import json
import pytest
from conftest import FixtureBackend
from voxstage.director import Director
from voxstage.store import exclusive

def test_incremental_cache_and_one_line_edit(tmp_path,scene,backend):
    first=Director(tmp_path,backend).run(scene)
    assert first['status']=='complete' and first['stats']['tts_calls']==2
    repeat=Director(tmp_path,backend).run(scene)
    assert repeat['stats']['tts_calls']==repeat['stats']['asr_calls']==0
    assert repeat['stats']['tts_cache_hits']==2
    edited=scene.model_copy(deep=True);edited.cues[0].text='不要打开右边的舱门。'
    revised=Director(tmp_path,backend).run(edited)
    assert revised['stats']['tts_calls']==1 and revised['stats']['tts_cache_hits']==1
    assert revised['run_id']!=first['run_id']

def test_timing_edit_does_not_regenerate_voice(tmp_path,scene,backend):
    Director(tmp_path,backend).run(scene)
    edited=scene.model_copy(deep=True);edited.cues[0].end=1.2
    result=Director(tmp_path,backend).run(edited)
    assert result['stats']['tts_calls']==0 and result['status']=='complete'

def test_budgeted_repair(tmp_path,scene):
    backend=FixtureBackend(bad_attempts=1)
    result=Director(tmp_path,backend).run(scene)
    assert result['status']=='complete'
    assert result['takes'][0]['decision']['action']=='resynthesize'
    assert result['takes'][1]['decision']['action']=='accept'
    assert backend.calls==3

def test_no_export_after_budget_exhaustion(tmp_path,scene):
    backend=FixtureBackend(bad_attempts=100)
    result=Director(tmp_path,backend).run(scene)
    assert result['status']=='needs_review' and len(result['takes'])==4
    assert result['rejected_cues']==['one','two'] and 'export' not in result
    assert not (tmp_path/'data/runs'/result['run_id']/'export/dialogue.wav').exists()

def test_controlled_interruption_reuses_verified_takes(tmp_path,scene,backend):
    with pytest.raises(RuntimeError,match='controlled interruption'):
        Director(tmp_path,backend).run(scene,interrupt_after_cue='one')
    resumed=Director(tmp_path,backend).run(scene)
    assert resumed['stats']['tts_calls']==1 and resumed['stats']['tts_cache_hits']==1

def test_corrupt_raw_cache_regenerates_audio(tmp_path,scene,backend):
    result=Director(tmp_path,backend).run(scene)
    take=result['takes'][0]
    (tmp_path/'data/cache/tts'/take['take_key']/'raw.wav').write_bytes(b'broken')
    resumed=Director(tmp_path,backend).run(scene)
    assert resumed['stats']['tts_calls']==1

def test_asr_reference_leakage_is_rejected(tmp_path,scene,backend):
    original=backend.transcribe
    def leak(**kwargs):return {**original(**kwargs),'reference_supplied':True}
    backend.transcribe=leak
    with pytest.raises(ValueError,match='reference'):Director(tmp_path,backend).run(scene)

def test_concurrent_writer_cannot_take_lock(tmp_path):
    path=tmp_path/'scene.lock'
    with exclusive(path):
        with pytest.raises(RuntimeError,match='another worker'):
            with exclusive(path):pass

def test_lock_releases_after_exception(tmp_path):
    path=tmp_path/'scene.lock'
    with pytest.raises(ValueError):
        with exclusive(path):raise ValueError('fail')
    with exclusive(path):pass

def test_missing_critical_word_rejected_even_with_low_cer(scene):
    from voxstage.director import diagnose
    from voxstage.text import compare
    cue=scene.cues[0].model_copy(deep=True)
    cue.text='请不要打开右边的舱门，因为外面还没有完成检查。';cue.protected_terms=['不要']
    observed=cue.text.replace('不要','要')
    content=compare(cue.text,observed)
    assert content['cer']<scene.policy.max_cer
    signal={'sample_rate':24000,'rms_dbfs':-20,'clipped_fraction':0,'duration':1}
    assert 'protected_term_missing' in diagnose(cue,scene.policy,signal,content)

def test_mutated_model_instance_is_revalidated(tmp_path,scene,backend):
    from pydantic import ValidationError
    scene.cues[0].end=3
    with pytest.raises(ValidationError):Director(tmp_path,backend).run(scene)
