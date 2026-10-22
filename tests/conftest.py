from pathlib import Path
import json
import os
import sys
import uuid
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'runtime/deps/common')]

if sys.platform=='win32' and os.environ.get('VOXSTAGE_TEST_TMP'):
    @pytest.fixture
    def tmp_path():
        folder=ROOT/'data/tests'/uuid.uuid4().hex
        folder.mkdir(parents=True,mode=0o755)
        return folder

@pytest.fixture
def scene():
    from voxstage.contracts import Scene
    return Scene.model_validate({'id':'test','title':'Original test fixture',
        'cast':[{'id':'actor','name':'Test actor','speaker':'Vivian'}],
        'cues':[{'id':'one','character':'actor','text':'不要打开舱门。','language':'Chinese',
                 'start':0,'end':1.5,'tone':'neutral'},
                {'id':'two','character':'actor','text':'Turn the blue key.','language':'English',
                 'start':2,'end':3.5,'tone':'neutral'}],
        'policy':{'max_attempts':2},'seed':42})


class FixtureBackend:
    """Test double with authored transcripts, never a real speech model."""
    label='synthetic-test-double'
    identity={'backend':'synthetic-test-double','version':1}
    def __init__(self,bad_attempts=0,duration=1.0):
        self.calls=0;self.transcripts={};self.current='';self.bad_attempts=bad_attempts;self.duration=duration
    def synthesize(self,*,text,language,speaker,instruct,seed,output):
        import numpy as np
        import hashlib
        from voxstage import audio
        self.calls+=1;self.current=text
        text_component=int(hashlib.sha256(text.encode()).hexdigest()[:4],16)
        values=.1*np.sin(2*np.pi*(200+(seed+text_component)%1800)*np.arange(round(audio.RATE*self.duration))/audio.RATE)
        audio.write(output,values)
        self.transcripts[audio.digest(output)]='wrong words' if self.calls<=self.bad_attempts else text
        return {'sample_rate':audio.RATE,'fixture':True,'elapsed_seconds':0}
    def transcribe(self,*,audio,language):
        from voxstage.audio import digest
        return {'text':self.transcripts.get(digest(audio),self.current),'language':language,
                'reference_supplied':False,'fixture_annotation':True}
    def close(self):pass

@pytest.fixture
def backend():return FixtureBackend()
