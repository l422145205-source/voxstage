"""Bounded model RPC; compatible environments communicate without importing each other."""
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Lock
import json
import subprocess
import sys
import uuid


class Worker:
    def __init__(self,root,kind,*,python=None,timeout=180):
        self.root=Path(root).resolve();self.kind=kind;self.timeout=timeout
        self.lock=Lock();self.queue=Queue();self.process=None;self.log=None;self.info=None
        environment=self.root/'runtime/venvs'/kind
        local_python=environment/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
        self.python=python or (str(local_python) if local_python.is_file() else sys.executable)

    def start(self):
        if self.process is not None:
            return
        logdir=self.root/'runtime/logs';logdir.mkdir(parents=True,exist_ok=True)
        self.log=(logdir/(self.kind+'-'+uuid.uuid4().hex+'.log')).open('w',encoding='utf-8')
        command=[self.python,'-X','utf8','-u','-B',str(self.root/'scripts/model_worker.py'),
            '--kind',self.kind,'--deps',str(self.root/'runtime/deps'/self.kind),
            '--common',str(self.root/'runtime/deps/common'),'--model',str(self.root/'runtime/models'/self.kind)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,encoding='utf-8',bufsize=1,shell=False)
        self.queue=Queue()
        stdout=self.process.stdout;queue=self.queue
        def reader():
            for line in stdout:
                queue.put(line)
            queue.put(None)
        Thread(target=reader,daemon=True).start()
        ready=self.receive()
        if ready.get('event')!='ready' or ready.get('kind')!=self.kind:
            self.close();raise RuntimeError('model did not report ready')
        self.info=ready

    def receive(self):
        try:
            line=self.queue.get(timeout=self.timeout)
        except Empty as exc:
            self.close();raise TimeoutError('model worker exceeded deadline') from exc
        if line is None:
            log=self.log.name if self.log else 'worker log'
            self.close();raise RuntimeError('model worker exited; inspect '+str(log))
        if len(line)>65536:
            self.close();raise ValueError('oversized worker response')
        result=json.loads(line)
        if result.get('version')!=1:
            raise ValueError('model protocol version mismatch')
        return result

    def call(self,method,**params):
        with self.lock:
            self.start()
            request={'version':1,'id':uuid.uuid4().hex,'method':method,**params}
            self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');self.process.stdin.flush()
            response=self.receive()
            if response.get('id')!=request['id']:
                self.close();raise ValueError('worker response request mismatch')
            if 'error' in response:
                raise RuntimeError(response['error'])
            return response['result']

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
            self.process=None
        if self.log:
            self.log.close()


class ModelBackend:
    label='qwen3-local'
    def __init__(self,root,python=None):
        self.root=Path(root).resolve()
        self.tts=Worker(root,'tts',python=python);self.asr=Worker(root,'asr',python=python)
        self.identity={'backend':self.label,'models':json.loads((self.root/'models.lock.json').read_text()),
            'tts_package':'0.1.1','asr_package':'0.0.6','sampling':{'temperature':0.8,'top_p':0.95,'max_new_tokens':1200}}

    def synthesize(self,*,text,language,speaker,instruct,seed,output):
        return self.tts.call('synthesize',text=text,language=language,speaker=speaker,instruct=instruct,seed=seed,output=str(output))

    def transcribe(self,*,audio,language):
        return self.asr.call('transcribe',audio=str(audio),language=language)

    def close(self):
        self.tts.close();self.asr.close()
