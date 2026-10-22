"""WAV contracts, acoustic checks, pitch-preserving timing, and source-bound exports."""
from pathlib import Path
import hashlib
import json
import subprocess
import wave
import numpy as np
from scipy.signal import resample_poly

RATE = 24000


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    with wave.open(str(path),'rb') as w:
        if w.getsampwidth()!=2 or w.getnchannels()!=1:
            raise ValueError('expected mono PCM16 WAV')
        rate=w.getframerate()
        values=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
    if not len(values) or len(values)>RATE*360:
        raise ValueError('empty or excessive audio')
    return values,rate


def write(path, values, rate=RATE):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('invalid waveform')
    if np.max(np.abs(values))>1:
        raise ValueError('waveform must be limited before export')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.partial.wav')
    with wave.open(str(temporary),'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.round(np.clip(values,-1,32767/32768)*32768).astype('<i2').tobytes())
    temporary.replace(path)


def analyze(path):
    values,rate=read(path)
    rms=float(np.sqrt(np.mean(values.astype(np.float64)**2)))
    peak=float(np.max(np.abs(values)))
    # Oversampled estimate, not an EBU-compliant true-peak meter.
    oversampled=float(np.max(np.abs(resample_poly(values,4,1))))
    return {'sample_rate':rate,'samples':len(values),'duration':len(values)/rate,
        'rms_dbfs':20*float(np.log10(max(rms,1e-10))),'sample_peak':peak,
        'oversampled_peak':oversampled,'clipped_fraction':float(np.mean(np.abs(values)>=0.999)),
        'sha256':digest(path)}


def speed_fit(source, destination, capacity, maximum):
    info=analyze(source)
    ratio=info['duration']/capacity
    if ratio>maximum+1e-6:
        raise ValueError('duration exceeds permitted time compression')
    if ratio<=1:
        values,rate=read(source);write(destination,values,rate)
        return {'speed':1.0,'method':'unaltered','source_sha256':info['sha256']}
    import imageio_ffmpeg
    # Slight margin absorbs container/sample rounding. No truncation is allowed.
    speed=min(maximum,ratio+0.002)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-y',
        '-i',str(source),'-af',f'atempo={speed:.8f}','-ac','1','-ar',str(RATE),'-c:a','pcm_s16le',str(destination)]
    result=subprocess.run(command,capture_output=True,text=True,shell=False,timeout=30)
    if result.returncode:
        raise RuntimeError('time-scale transform failed: '+result.stderr[-1000:])
    if analyze(destination)['duration']>capacity+1/RATE:
        raise ValueError('time-scale output still exceeds cue slot')
    return {'speed':speed,'method':'ffmpeg_atempo','source_sha256':info['sha256']}


def stamp(seconds):
    ms=round(seconds*1000); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def export_scene(scene, accepted, folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if set(accepted)!={c.id for c in scene.cues}:
        raise ValueError('cannot publish scene with missing or rejected cues')
    output=np.zeros(round(max(c.end for c in scene.cues)*RATE),np.float32)
    manifest=[]; subtitles=[]
    for i,cue in enumerate(scene.cues,1):
        record=accepted[cue.id]; path=Path(record['path'])
        if digest(path)!=record['sha256']:
            raise ValueError('accepted audio contract/hash changed')
        values,rate=read(path)
        if rate!=RATE:raise ValueError('accepted audio sample rate changed')
        start=round(cue.start*RATE); end=start+len(values)
        if end>round(cue.end*RATE) or end>len(output):
            raise ValueError('accepted cue exceeds export bounds')
        output[start:end]+=values
        subtitles.append(f'{i}\n{stamp(cue.start)} --> {stamp(min(cue.end,cue.start+len(values)/rate))}\n{cue.text}\n')
        manifest.append({'cue_id':cue.id,'character':cue.character,'start':cue.start,
                         'end':cue.start+len(values)/rate,'take_sha256':record['sha256'],
                         'text':cue.text,'speech_text':cue.spoken,'source_take_key':record['take_key']})
    # A single shared gain preserves inter-line relative levels; never clip.
    peak=float(np.max(np.abs(output))); gain=min(1.0,0.95/max(peak,1e-8));output*=gain
    write(folder/'dialogue.wav',output)
    (folder/'dialogue.srt').write_text('\n'.join(subtitles),encoding='utf-8')
    result={'scene_id':scene.id,'audio':analyze(folder/'dialogue.wav'),'gain':gain,
            'cues':manifest,'subtitle_timing':'cue-level from placed audio, not word/phoneme alignment'}
    (folder/'timeline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
