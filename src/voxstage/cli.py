from pathlib import Path
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description='VoxStage dialogue production')
    parser.add_argument('--root',type=Path,default=Path.cwd())
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('scene',type=Path)
    serve=sub.add_parser('serve');serve.add_argument('--models',action='store_true');serve.add_argument('--port',type=int,default=8877)
    args=parser.parse_args();root=args.root.resolve()
    backend=None
    if args.command=='run' or args.models:
        from .workers import ModelBackend
        backend=ModelBackend(root)
    if args.command=='serve':
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(root,backend),host='127.0.0.1',port=args.port)
    else:
        from .contracts import Scene
        from .director import Director
        try:
            scene=Scene.model_validate_json(args.scene.read_text(encoding='utf-8'))
            result=Director(root,backend,lambda e:print(json.dumps(e,ensure_ascii=False),flush=True)).run(scene)
            print(json.dumps({k:result[k] for k in ['run_id','status','stats','rejected_cues']},ensure_ascii=False,indent=2))
        finally:backend.close()


if __name__=='__main__':main()
