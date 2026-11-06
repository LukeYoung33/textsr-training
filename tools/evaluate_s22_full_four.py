"""Read-only latest Full paired/mix30/identity/guard probes on Mojito049-052."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util
import json
import shutil
import sys
import time

DATA=Path('/AI_DNS/guang.yang/TextSR/data')
BASE=DATA/'v3pre_three_mojito280_20261028'
RUN=DATA/'full_s22_training_matrix_20261028'
OUT=DATA/'s22_full_four_mojito049052_20261029'
sys.path.insert(0,str(BASE/'_build'))


def main():
    """@luna-comment Input:declared four Full configs/latest numeric weights. Output:16 frozen native residual/RGB records. Key Steps:verify config hashes;copy stable weights;reuse original capture;check inputs and residual branch. Key Params:256/224/batch8/crop8/smooth16. State/Dependencies:read-only live training,owned output,CUDA selected externally."""
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.build_camera_2x2_viewer import digest
    helper=BASE/'diagnostics/residual_frequency_049052_20261028/analyze.py'
    spec=importlib.util.spec_from_file_location('s22_residual_capture',helper)
    capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'BINDINGS.json').exists(),'Existing immutable probe'
    catalog=json.loads((RUN/'CONFIG_BINDINGS.json').read_text())['runs'];bindings=[]
    for strategy in ('paired','gt_degrade30','identity','guard'):
        key=strategy+'_full';declared=catalog[key]
        assert declared['arm']=='full' and declared['strategy']==strategy
        config=Path(declared['config']);assert digest(config)==declared['sha256']
        cfg=yaml.safe_load(config.read_text());assert cfg['train']['sdf_boundary_weight']==0
        directory=RUN/'training/experiments'/declared['name']/'models'
        files=[p for p in directory.glob('net_g_*.pth') if p.stem[6:].isdigit()]
        checkpoint=max(files,key=lambda p:int(p.stem[6:]));iteration=int(checkpoint.stem[6:])
        expected=digest(checkpoint);frozen=OUT/'checkpoints'/f'{key}_{iteration}.pth';frozen.parent.mkdir(exist_ok=True)
        shutil.copy2(checkpoint,frozen);assert digest(frozen)==expected==digest(checkpoint)
        frozen_config=OUT/'configs'/f'{key}.yml';frozen_config.parent.mkdir(exist_ok=True);shutil.copy2(config,frozen_config)
        bindings.append(dict(key=key,strategy=strategy,iteration=iteration,checkpoint=str(frozen),checkpoint_sha256=expected,source_checkpoint=str(checkpoint),config=str(frozen_config),config_sha256=digest(frozen_config),frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    (OUT/'BINDINGS.json').write_text(json.dumps(bindings,indent=2))
    entries={e['id']:e for e in json.loads((BASE/'selection.json').read_text())['samples']};records=[]
    for binding in bindings:
        cfg=yaml.safe_load(Path(binding['config']).read_text());model=build_network(cfg['network_g']).cuda().eval()
        model.load_state_dict(torch.load(binding['checkpoint'],map_location='cpu')['params'],strict=True)
        for identity,display_roi in capture.ROIS.items():
            started=time.time();entry=entries[identity]
            assert digest(entry['input'])==entry['input_sha256'] and entry['orientation']==6
            source=cv2.cvtColor(cv2.imread(entry['input'],cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
            xa,ya,xb,yb=display_roi;h=source.shape[0];roi=[ya,h-xb,yb,h-xa]
            result=capture.capture(source,model,roi)
            output=np.rot90(result.pop('replay'),k=-1)
            path=OUT/f'{identity}_{binding["key"]}_output.png'
            assert cv2.imwrite(str(path),cv2.cvtColor(output,cv2.COLOR_RGB2BGR))
            arrays={k:np.rot90(result.pop(k),k=-1) for k in ('x','r','xp','rp','center')}
            artifact=OUT/f'{identity}_{binding["key"]}_raw_roi.npz';np.savez_compressed(artifact,**arrays)
            record=dict(id=identity,key=binding['key'],iteration=binding['iteration'],input_sha256=entry['input_sha256'],display_roi=display_roi,artifact_sha256=digest(artifact),output_sha256=digest(path),seconds=time.time()-started,**result)
            records.append(record)
            (OUT/'RESULT.json').write_text(json.dumps(dict(complete=False,records=records,bindings=bindings),indent=2))
            print(json.dumps(dict(id=identity,key=binding['key'],iteration=binding['iteration'],done=len(records),total=16,seconds=record['seconds'])),flush=True)
        del model;torch.cuda.empty_cache()
    (OUT/'RESULT.json').write_text(json.dumps(dict(complete=True,records=records,bindings=bindings,capture_sha256=digest(helper),runner_sha256=digest(__file__),policy='Same four frozen inputs/native grid;actual checkpoint iterations differ;no training changes or subjective quality verdict.'),indent=2))


if __name__=='__main__':main()
