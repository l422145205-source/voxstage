import numpy as np
import pytest
from voxstage import audio

def tone(path,duration=1,amplitude=.2):
    audio.write(path,amplitude*np.sin(2*np.pi*440*np.arange(round(duration*audio.RATE))/audio.RATE))

def test_pcm_roundtrip_and_analysis(tmp_path):
    path=tmp_path/'tone.wav';tone(path)
    info=audio.analyze(path)
    assert info['duration']==1 and info['sample_rate']==24000
    assert info['clipped_fraction']==0 and -20<info['rms_dbfs']<-10
    assert info['sha256']==audio.digest(path)

@pytest.mark.parametrize('values',[[float('nan')],[float('inf')],[1.2],[],[[.1,.2]]])
def test_invalid_waveforms_fail_closed(tmp_path,values):
    with pytest.raises(ValueError):audio.write(tmp_path/'invalid.wav',values)

def test_pitch_preserving_time_compression(tmp_path):
    source=tmp_path/'source.wav';tone(source,1.1)
    dest=tmp_path/'fit.wav';result=audio.speed_fit(source,dest,1,1.15)
    values,rate=audio.read(dest)
    frequency=np.fft.rfftfreq(len(values),1/rate)[np.argmax(np.abs(np.fft.rfft(values)))]
    assert result['method']=='ffmpeg_atempo'
    assert audio.analyze(dest)['duration']<=1.03
    assert abs(frequency-440)<8
    assert audio.analyze(source)['duration']==1.1

def test_never_truncate_to_fit(tmp_path):
    source=tmp_path/'source.wav';tone(source,1.3)
    with pytest.raises(ValueError):audio.speed_fit(source,tmp_path/'out.wav',1,1.15)
    assert not (tmp_path/'out.wav').exists()

def test_export_provenance_and_bounds(tmp_path,scene):
    path=tmp_path/'tone.wav';tone(path)
    accepted={c.id:{'path':str(path),'sha256':audio.digest(path),'take_key':'a'*64} for c in scene.cues}
    report=audio.export_scene(scene,accepted,tmp_path/'export')
    assert report['audio']['duration']==3.5
    assert len(report['cues'])==2
    path.write_bytes(b'corrupted')
    with pytest.raises((ValueError,EOFError)):audio.export_scene(scene,accepted,tmp_path/'broken')

def test_missing_cue_blocks_export(tmp_path,scene):
    with pytest.raises(ValueError):audio.export_scene(scene,{},tmp_path/'export')

def test_sub_frame_slot_overflow_does_not_mix_over_next_cue(tmp_path,scene):
    path=tmp_path/'tone.wav';tone(path,1.51)
    accepted={c.id:{'path':str(path),'sha256':audio.digest(path),'take_key':'a'*64} for c in scene.cues}
    with pytest.raises(ValueError,match='bounds'):audio.export_scene(scene,accepted,tmp_path/'export')
