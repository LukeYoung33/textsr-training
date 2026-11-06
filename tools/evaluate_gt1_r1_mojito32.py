"""Freeze current four GT1/R1 checkpoints and reuse accepted fixed32 evaluator."""
from pathlib import Path
import argparse
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import evaluate_five_controls_mojito32 as core
from tools.run_uniform20_steep_scurve import save,digest
SOURCE=core.DATA/'gt1_r1_mixture_four_500k_20261104'
core.OUT=core.DATA/'gt1_r1_four_latest_mojito32_20261105'
core.ARMS={k:v['label'] for k,v in json.loads((SOURCE/'CONFIG_BINDINGS.json').read_text())['runs'].items()}
core.__file__=__file__

def prepare():
    """@luna-comment Input:four live experiments. Output:immutable latest complete numeric checkpoint bindings. Key Steps:verify receipts/hash before and after copy,freeze configs and fixed32. Key Params:exclude mutable latest alias. State/Dependencies:read-only training files,owned evaluation root."""
    import yaml
    out=core.OUT
    (out/'models').mkdir(exist_ok=True)
    bindings={}
    for arm,b in json.loads((SOURCE/'CONFIG_BINDINGS.json').read_text())['runs'].items():
        config=Path(b['config']); assert digest(config)==b['config_sha256']
        cfg=yaml.safe_load(config.read_text())
        folder=SOURCE/'experiments'/cfg['name']/'models'
        candidates=sorted(((int(p.stem.split('_')[-1]),p) for p in folder.glob('net_g_*.pth') if p.stem.split('_')[-1].isdigit()),reverse=True)
        chosen=None
        for iteration,checkpoint in candidates:
            receipt=folder.parent/'training_states'/f'{iteration}.verified.json'
            if receipt.exists():
                chosen=(iteration,checkpoint,receipt);break
        assert chosen is not None,(arm,str(folder))
        iteration,checkpoint,receipt=chosen
        verified=json.loads(receipt.read_text())
        before=digest(checkpoint)
        assert verified['iter']==iteration and verified['model_sha256']==before and verified['model']==str(checkpoint),(arm,verified)
        model=out/'models'/(arm+'.pth'); conf=out/'models'/(arm+'.yml')
        shutil.copy2(checkpoint,model);shutil.copy2(config,conf)
        assert digest(model)==before==digest(checkpoint)
        assert digest(conf)==b['config_sha256']
        bindings[arm]=dict(config=str(conf),config_sha256=digest(conf),checkpoint=str(model),checkpoint_sha256=before,source_checkpoint=str(checkpoint),iteration=iteration,label=b['label'],stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    shutil.copy2(core.DATA/'mojito280_subset32_20261104/selection.json',out/'selection.json')
    assert len(json.loads((out/'selection.json').read_text())['samples'])==32
    save(out/'MODEL_BINDINGS.json',bindings)
    save(out/'RUN_BINDING.json',dict(selection_sha256=digest(out/'selection.json'),post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'),outputs=256))

core.prepare=prepare
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('--arm',choices=list(core.ARMS));args=parser.parse_args()
    core.worker(args.arm) if args.mode=='worker' else core.run()
