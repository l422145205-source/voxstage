from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'runtime/deps/common')]
from voxstage.workers import ModelBackend

def main():
    destination=ROOT/'data/smoke/hello.wav';destination.parent.mkdir(parents=True,exist_ok=True)
    backend=ModelBackend(ROOT)
    try:
        result=backend.synthesize(text='指挥官，飞船已经准备好了。',language='Chinese',speaker='Vivian',
            instruct='清楚、自然地说话。',seed=2026,output=destination)
        print('TTS',json.dumps(result,ensure_ascii=False),flush=True)
        recognition=backend.transcribe(audio=destination,language='Chinese')
        print('ASR',json.dumps(recognition,ensure_ascii=False),flush=True)
        report={'synthesis':result,'recognition':recognition,'workers':{'tts':backend.tts.info,'asr':backend.asr.info}}
        (ROOT/'reports').mkdir(exist_ok=True)
        (ROOT/'reports/model_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:backend.close()

if __name__=='__main__':main()
