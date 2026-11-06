"""Frozen latest no-SDF versus same-iteration SDF native residual probes."""
from pathlib import Path
import importlib.util
import json
import shutil
import sys
import time
import argparse

DATA=Path('/AI_DNS/guang.yang/TextSR/data')
BASE=DATA/'v3pre_three_mojito280_20261028'
RUN=DATA/'v3pre_v3m_nosdf_250k_20261028'
OLD=DATA/'v3pre_v3m_three_20261027/pretrain'
OUT=DATA/'nosdf_latest_residual_049052_20261028'
sys.path.insert(0,str(BASE/'_build'))


def main():
    """@luna-comment Input:owned output name,running no-SDF checkpoints/frozen inputs. Output:24 native residual records. Key Steps:freeze complete numeric checkpoints; strict load; accepted full-frame capture; hash and branch audit. Key Params:original256/224/crop8/smooth16. State/Dependencies:GPU3; read-only training jobs and owned output only."""
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name',default=OUT.name)
    args=parser.parse_args()
    assert Path(args.output_name).name==args.output_name
    OUT=DATA/args.output_name
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.build_camera_2x2_viewer import digest
    helper=BASE/'diagnostics/residual_frequency_049052_20261028/analyze.py'
    spec=importlib.util.spec_from_file_location('frozen_residual_capture',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'BINDINGS.json').exists(),'Refusing to overwrite frozen probe'
    bindings=[]
    for arm in ('full','text15','adapt_l1'):
        directory=RUN/f'training/experiments/v3pre_nosdf_{arm}_20261028/models'
        candidates=[p for p in directory.glob('net_g_*.pth') if p.stem[6:].isdigit()]
        source=max(candidates,key=lambda p:int(p.stem[6:]))
        iteration=int(source.stem[6:]);before=digest(source)
        for kind,checkpoint,config in [('nosdf',source,RUN/f'training/configs/{arm}.yml'),('sdf_same',OLD/f'training/experiments/v3pre_pretrain_{arm}_20261027/models/net_g_{iteration}.pth',OLD/f'training/configs/{arm}.yml')]:
            frozen=OUT/'checkpoints'/f'{kind}_{arm}_{iteration}.pth';frozen.parent.mkdir(exist_ok=True)
            expected=digest(checkpoint);shutil.copy2(checkpoint,frozen)
            assert digest(frozen)==expected and digest(checkpoint)==expected
            cfg=yaml.safe_load(config.read_text())
            if kind=='nosdf':assert cfg['train']['sdf_boundary_weight']==0
            bindings.append(dict(key=f'{kind}_{arm}',arm=arm,kind=kind,iteration=iteration,source=str(checkpoint),checkpoint=str(frozen),sha256=expected,config=str(config),config_sha256=digest(config),sdf_weight=cfg['train']['sdf_boundary_weight']))
        assert digest(source)==before
    (OUT/'BINDINGS.json').write_text(json.dumps(bindings,indent=2))
    entries={e['id']:e for e in json.loads((BASE/'selection.json').read_text())['samples']}
    records=[]
    for binding in bindings:
        config=yaml.safe_load(Path(binding['config']).read_text())
        model=build_network(config['network_g']).cuda().eval()
        model.load_state_dict(torch.load(binding['checkpoint'],map_location='cpu')['params'],strict=True)
        for identity,display_roi in module.ROIS.items():
            started=time.time();entry=entries[identity]
            assert digest(entry['input'])==entry['input_sha256'] and entry['orientation']==6
            source=cv2.cvtColor(cv2.imread(entry['input'],cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
            xa,ya,xb,yb=display_roi;h=source.shape[0];roi=[ya,h-xb,yb,h-xa]
            result=module.capture(source,model,roi)
            output=np.rot90(result.pop('replay'),k=-1)
            cv2.imwrite(str(OUT/f'{identity}_{binding["key"]}_output.png'),cv2.cvtColor(output,cv2.COLOR_RGB2BGR))
            arrays={k:np.rot90(result.pop(k),k=-1) for k in ('x','r','xp','rp','center')}
            artifact=OUT/f'{identity}_{binding["key"]}_raw_roi.npz'
            np.savez_compressed(artifact,**arrays)
            x=arrays['x'];r=arrays['r'];ink=(cv2.dilate(x,np.ones((15,15),np.uint8))-x)>45
            core=cv2.erode(ink.astype(np.uint8),np.ones((3,3),np.uint8))>0
            ring=(cv2.dilate(ink.astype(np.uint8),np.ones((7,7),np.uint8))>0)&~ink
            signed={name:dict(count=int(mask.sum()),mean=float(r[mask].mean())) for name,mask in [('core',core),('ring',ring)]}
            record=dict(id=identity,key=binding['key'],iteration=binding['iteration'],input_sha256=entry['input_sha256'],display_roi=display_roi,signed_regions=signed,artifact_sha256=digest(artifact),seconds=time.time()-started,**result)
            records.append(record)
            (OUT/'RESULT.json').write_text(json.dumps(dict(complete=False,records=records,bindings=bindings),indent=2))
            print(json.dumps(dict(id=identity,key=binding['key'],iteration=binding['iteration'],done=len(records),total=24,seconds=record['seconds'])),flush=True)
        del model;torch.cuda.empty_cache()
    (OUT/'RESULT.json').write_text(json.dumps(dict(complete=True,records=records,bindings=bindings,capture_sha256=digest(helper),runner_sha256=digest(__file__),policy='Fixed original inputs and native grid; descriptive four-image probe, not population quality or controlled historical causality.'),indent=2))


if __name__=='__main__':main()
