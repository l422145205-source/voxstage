"""Public allowlist, private-path redaction, source ZIP, and Git object hashes."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TOP=['README.md','pyproject.toml','requirements-tts.txt','requirements-asr.txt',
     'models.lock.json','LICENSE','THIRD_PARTY.md','.gitignore']
REPORTS=['model_smoke.json','production_pilot.json','protocol_freeze.json',
         'final_constraints_replay.json','model_provenance.json','browser_validation.json','test-final.txt']
TEXT={'.py','.md','.toml','.txt','.json','.js','.css','.html','.yml'}


def main():
    stage=ROOT/'dist/public';stage.mkdir(parents=True,exist_ok=True)
    files=[ROOT/p for p in TOP]+[ROOT/'reports'/p for p in REPORTS]
    for folder in ['src','scripts','tests','docs','examples','.github']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix!='.pyc')
    entries=[]
    for source in sorted(set(files)):
        relative=source.relative_to(ROOT).as_posix();data=source.read_bytes()
        if source.suffix in TEXT or source.name in TOP:
            value=data.decode('utf-8-sig')
            for root in [str(ROOT),str(ROOT).replace('\\','/'),str(ROOT).replace('\\','\\\\')]:
                value=value.replace(root,'${PROJECT_ROOT}')
            forbidden=r'(?i)(?:[A-Z]:[\\/]+Users[\\/]+[^\\/\s]+|(?<![\w.])1[3-9]\d{9}(?![\w.])|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20})'
            if re.search(forbidden,value):raise ValueError('private marker: '+relative)
            data=value.encode('utf-8')
        if len(data)>2_000_000:raise ValueError('unexpected large public file '+relative)
        target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        entries.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                        'git_blob_sha':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()})
    manifest={'files':entries,'excluded':['model weights','dependencies','local logs','private resumes','credentials']}
    (ROOT/'dist/publication_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=ROOT/'dist/voxstage-source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for entry in entries:z.write(stage/entry['path'],'voxstage/'+entry['path'])
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'files':len(entries),'public_bytes':sum(x['bytes'] for x in entries),
                      'zip_bytes':archive.stat().st_size,'zip_crc_verified':True}))


if __name__=='__main__':main()
