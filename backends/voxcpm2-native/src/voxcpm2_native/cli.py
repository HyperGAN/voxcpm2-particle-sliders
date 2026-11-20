"""Explicit product dispatch; shared core owns updates, FULL/RAW and inference."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    train=sub.add_parser('train');train.add_argument('--config',type=Path,required=True);train.add_argument('--out',type=Path,required=True)
    train.add_argument('--resume',type=Path);train.add_argument('--stop-after',type=int)
    render=sub.add_parser('render');render.add_argument('--artifact',type=Path,required=True);render.add_argument('--options',type=Path,required=True)
    render.add_argument('--request',type=Path,required=True);render.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():p.error('fresh output path required; never overwrite a prior attempt')
    if a.mode=='train':
        from particle_sliders.lifecycle import SessionConfig,atomic_json
        from .factory import open_training
        config=SessionConfig.from_dict(json.loads(a.config.read_text()))
        if a.stop_after is not None and a.stop_after<0:p.error('stop-after must be nonnegative')
        a.out.mkdir(parents=True,exist_ok=False)
        with open_training(config,resume=a.resume)as session:
            session.train(stop_after=a.stop_after);session.checkpoint(a.out/'FULL.pt')
            session.export_inference(a.out/'RAW.pt',selection='raw')
            atomic_json(a.out/'run.json',dict(config=config.as_dict(),completed_steps=session.game.completed_steps,
                resumed_from=None if a.resume is None else str(a.resume),selection='raw',weight_EMA=None))
    else:
        from particle_sliders.inference import open_inference
        from .factory import INFERENCE
        options=json.loads(a.options.read_text());request=json.loads(a.request.read_text())
        with open_inference(a.artifact,options=options,backend=INFERENCE,operation='render',output_type='audio',native_request=request)as session:
            session.render(**request).save(a.out)

if __name__=='__main__':main()
