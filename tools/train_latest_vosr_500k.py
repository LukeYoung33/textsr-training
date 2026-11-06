"""Isolated latest-VOSR scratch controls and exact absolute LR schedule."""
from pathlib import Path
import sys,argparse,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def rate(iteration):
    """@luna-comment Input:training update number. Output:absoluteLR. Key Steps:four explicit boundaries. Key Params:1e-4 to5e-6,500K. State/Dependencies:pure function."""
    value=1e-4
    for boundary,target in [(200000,5e-5),(300000,2.5e-5),(400000,1e-5),(450000,5e-6)]:
        if iteration>=boundary:value=target
    return value


def bootstrap():
    """@luna-comment Input:isolated runtime. Output:S22 registrations plus task-only scheduler. Key Steps:delegate historical bootstrap;dispatch only own schedule type. Key Params:BasicSR steps before update exceptiter1. State/Dependencies:process-local patch,old processes untouched."""
    from tools.train_full_s22_strategy import bootstrap as original
    original()
    from basicsr.models.base_model import BaseModel
    from torch.optim.lr_scheduler import LambdaLR
    previous=BaseModel.setup_schedulers

    def setup(model):
        """@luna-comment Input:model optimizers. Output:exact absolute scheduler. Key Steps:delegate other types;map scheduler epoch+1 to update number. Key Params:single initial1e-4. State/Dependencies:owned training process."""
        if model.opt['train']['scheduler']['type']!='LatestVOSR500K':return previous(model)
        for optimizer in model.optimizers:
            assert all(abs(g['lr']-1e-4)<1e-12 for g in optimizer.param_groups)
            model.schedulers.append(LambdaLR(optimizer,lambda epoch:rate(epoch+1)/1e-4))
    BaseModel.setup_schedulers=setup


def smoke(config,output):
    """@luna-comment Input:full config and owned receipt. Output:actual9CUDA updates with exact seed snapshot and objective checks. Key Steps:select filtered patch,strictload,finite/changed weights,postramp checks. Key Params:L1 total equality;guard controlled cancellation probe. State/Dependencies:accepted baseline smoke implementation with isolated assertion policy."""
    import tools.train_full_s22_strategy as original
    import inspect
    text=inspect.getsource(original.smoke)
    old="    assert logs['uv_strong_scale']>0";assert text.count(old)==1
    text=text.replace(old,"    if opt['train'].get('uv_strong_weight',0)>0:assert logs['uv_strong_scale']>0\n    else:\n        assert model.strategy=='paired' and not model.calibration\n        assert abs(logs['l_total']-logs['l_native'])<1e-7\n        assert all(v==0 for k,v in logs.items() if k.endswith('_scale'))")
    namespace=dict(original.__dict__);exec(compile(text,'<latest_vosr_smoke>','exec'),namespace);namespace['smoke'](config,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');args,rest=parser.parse_known_args();bootstrap()
    if args.smoke_config:smoke(Path(args.smoke_config),Path(args.smoke_output))
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline;train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
