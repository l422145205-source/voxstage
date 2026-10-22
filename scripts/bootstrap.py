"""Download official pinned model snapshots; install dependencies only inside this project."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'tts': ('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', 'tts'),
    'asr': ('Qwen/Qwen3-ASR-0.6B', 'asr'),
}
ENDPOINT = os.environ.get('VOXSTAGE_MODEL_ENDPOINT', 'https://modelscope.cn').rstrip('/')


def session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504])))
    return s


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download_file(repo, revision, item, folder):
    name = item['path']
    target = folder/name
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    size = item['size']
    expected = item.get('lfs', {}).get('oid')
    if target.exists() and target.stat().st_size == size and (not expected or sha(target) == expected):
        return {'path': name, 'bytes': size, 'sha256': sha(target)}
    temporary = target.with_name(target.name+'.partial')
    for attempt in range(4):
        offset = temporary.stat().st_size if temporary.exists() else 0
        url = (f'{ENDPOINT}/models/{repo}/resolve/{revision}/{name}' if 'modelscope.cn' in ENDPOINT
               else f'{ENDPOINT}/{repo}/resolve/{revision}/{name}')
        with session().get(url,
                headers={'Range': f'bytes={offset}-'} if offset else {}, stream=True,
                timeout=(25, 90)) as response:
            response.raise_for_status()
            append = response.status_code == 206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
            with temporary.open('ab' if append else 'wb') as f:
                for block in response.iter_content(1024*1024):
                    if block:
                        f.write(block)
        if temporary.stat().st_size == size and (not expected or sha(temporary) == expected):
            temporary.replace(target)
            print('verified', repo, name, size, flush=True)
            return {'path': name, 'bytes': size, 'sha256': sha(target)}
        temporary.unlink(missing_ok=True)
    raise ValueError('download checksum/size mismatch: '+name)


def model(which):
    repo, label = MODELS[which]
    lock = ROOT/'models.lock.json'
    pins = json.loads(lock.read_text()) if lock.exists() else {}
    revision = pins.get(which, {}).get('revision')
    provider = 'modelscope' if 'modelscope.cn' in ENDPOINT else 'huggingface'
    pinned_provider = pins.get(which, {}).get('source', 'huggingface')
    if revision and pinned_provider != provider:
        raise ValueError(
            f'{which} is pinned to {pinned_provider}; endpoint is {provider}. '
            'Provider revisions are not interchangeable. Use the default ModelScope '
            'endpoint or supply a separately verified provider-specific model lock.')
    if 'modelscope.cn' in ENDPOINT:
        response = session().get(f'{ENDPOINT}/api/v1/models/{repo}/repo/files',
            params={'Revision':revision or 'master','Recursive':'true'},timeout=40)
        response.raise_for_status()
        files = response.json()['Data']['Files']
        if not revision:
            revision = next(x['Revision'] for x in files if x['Path']=='model.safetensors')
            pins[which] = {'repo':repo,'revision':revision,'source':'modelscope'}
            lock.write_text(json.dumps(pins,indent=2),encoding='utf-8')
        selected = [{'path':x['Path'],'size':x['Size'],'type':'file','lfs':{'oid':x['Sha256']}}
                    for x in files if x['Type']=='blob' and x['Path'].endswith(('.json','.safetensors','.txt','.model','.jinja'))]
        folder=ROOT/'runtime/models'/label;folder.mkdir(parents=True,exist_ok=True,mode=0o755)
        print('snapshot',which,revision,'bytes',sum(x['size'] for x in selected),flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(download_file,repo,revision,x,folder) for x in selected]
            records=[f.result() for f in as_completed(futures)]
        (folder/'download_manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,
            'source':'modelscope','files':records},indent=2),encoding='utf-8')
        return
    if not revision:
        metadata = session().get(f'{ENDPOINT}/api/models/{repo}', timeout=40)
        metadata.raise_for_status()
        revision = metadata.json()['sha']
        pins[which] = {'repo': repo, 'revision': revision}
        lock.write_text(json.dumps(pins, indent=2), encoding='utf-8')
    response = session().get(f'{ENDPOINT}/api/models/{repo}/tree/{revision}',
                            params={'recursive': 'true'}, timeout=40)
    response.raise_for_status()
    # Include the TTS tokenizer subdirectory, but never training/optimizer files.
    selected = [x for x in response.json() if x.get('type') == 'file'
        and x['path'].endswith(('.json','.safetensors','.txt','.model','.jinja'))
        and not x['path'].startswith(('optimizer','training'))]
    folder = ROOT/'runtime/models'/label
    folder.mkdir(parents=True, exist_ok=True, mode=0o755)
    print('snapshot', which, revision, 'bytes',sum(x['size'] for x in selected), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_file, repo, revision, item, folder) for item in selected]
        files = [f.result() for f in as_completed(futures)]
    (folder/'download_manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,
        'canonical_url':f'https://huggingface.co/{repo}/tree/{revision}','endpoint':ENDPOINT,
        'files':files},indent=2),encoding='utf-8')


def dependencies(which):
    # Two model workers intentionally have separate import roots. Global packages
    # and other projects are never upgraded. Torch must be installed for the GPU.
    package = {'tts':'qwen-tts==0.1.1','asr':'qwen-asr==0.0.6'}[which]
    version = {'tts':'4.57.3','asr':'4.57.6'}[which]
    folder = ROOT/'runtime/deps'/which
    folder.mkdir(parents=True, exist_ok=True, mode=0o755)
    common = ['soundfile==0.13.1','librosa==0.11.0','accelerate==1.12.0',
              'sox==1.5.0','einops==0.8.1','soxr==1.0.0','lazy-loader==0.4']
    extra = ['onnxruntime'] if which=='tts' else ['qwen-omni-utils','nagisa==0.2.11','soynlp==0.0.493']
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
        '--no-deps','--target',str(folder),package,'transformers=='+version,*common,*extra],check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind',choices=['model','deps'])
    parser.add_argument('worker',choices=['tts','asr'])
    args = parser.parse_args()
    (model if args.kind=='model' else dependencies)(args.worker)
