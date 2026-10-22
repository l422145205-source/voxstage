"""One model per import environment. Stdout is reserved for versioned JSON lines."""
from pathlib import Path
import argparse
import contextlib
import json
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['tts','asr'],required=True)
    parser.add_argument('--deps',type=Path,required=True)
    parser.add_argument('--common',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    sys.path[:0]=[str(args.deps),str(args.common)]
    protocol=sys.stdout
    started=time.perf_counter()
    # Upstream optional-dependency notices must not corrupt the protocol stream.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import transformers
        torch.set_num_threads(4)
        if args.kind=='tts':
            from qwen_tts import Qwen3TTSModel
            model=Qwen3TTSModel.from_pretrained(str(args.model),device_map=args.device,
                dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        else:
            from qwen_asr import Qwen3ASRModel
            model=Qwen3ASRModel.from_pretrained(str(args.model),dtype=torch.bfloat16,
                device_map=args.device,max_inference_batch_size=1,max_new_tokens=256,
                attn_implementation='sdpa',local_files_only=True)
    print(json.dumps({'version':1,'event':'ready','kind':args.kind,
        'load_seconds':time.perf_counter()-started,'transformers':transformers.__version__}),file=protocol,flush=True)
    for line in sys.stdin:
        request=None
        try:
            request=json.loads(line)
            if request.get('version')!=1 or not isinstance(request.get('id'),str):
                raise ValueError('invalid protocol envelope')
            started=time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
                if args.kind=='tts':
                    if request['method']!='synthesize': raise ValueError('unsupported TTS method')
                    import soundfile as sf
                    torch.manual_seed(request['seed'])
                    torch.cuda.manual_seed_all(request['seed'])
                    wavs,rate=model.generate_custom_voice(text=request['text'],
                        language=request['language'],speaker=request['speaker'],instruct=request['instruct'],
                        non_streaming_mode=True,max_new_tokens=1200,
                        do_sample=True,temperature=0.8,top_p=0.95)
                    destination=Path(request['output']);destination.parent.mkdir(parents=True,exist_ok=True)
                    sf.write(str(destination),wavs[0],rate,subtype='PCM_16')
                    result={'path':str(destination),'sample_rate':rate}
                else:
                    if request['method']!='transcribe': raise ValueError('unsupported ASR method')
                    # Never accept reference/expected text or contextual hints.
                    allowed={'version','id','method','audio','language'}
                    if set(request)-allowed: raise ValueError('ASR accepts audio and language only')
                    import soundfile as sf
                    values,rate=sf.read(request['audio'],dtype='float32')
                    results=model.transcribe(audio=(values,rate),language=request['language'],context='')
                    result={'text':results[0].text,'language':results[0].language,'reference_supplied':False}
            result['elapsed_seconds']=time.perf_counter()-started
            result['gpu_allocated_bytes']=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            print(json.dumps({'version':1,'id':request['id'],'result':result},ensure_ascii=False),file=protocol,flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'version':1,'id':request.get('id') if request else None,
                'error':type(exc).__name__+': '+str(exc)[:1000]},ensure_ascii=False),file=protocol,flush=True)


if __name__=='__main__':
    main()
