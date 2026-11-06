"""Paired random-input effect comparison; does not assert codec bit equality."""
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from realesrgan import lenovo_atomic as ref
from realesrgan.lenovo_degradation import LenovoDegradationPipeline
from check_lenovo_atomic_equivalence import statistics, sha


def main():
    torch.set_num_threads(4)
    out=ROOT/'datasets/lenovo_torch_equivalence_v1'; out.mkdir(parents=True,exist_ok=True)
    pipeline=LenovoDegradationPipeline(); records=[]
    mapping={'c1a':'gaussian','c1b':'motion','c1c':'zoom','c1d':'defocus'}
    for index in range(3):
        gt=cv2.imread(str(ROOT/f'datasets/lenovo_atomic_equivalence_v1/gt_{index}.png'))
        x=torch.from_numpy(gt[:,:,::-1].copy()).permute(2,0,1)[None].float()/255
        for key,(_,func) in ref.CATEGORIES.items():
            if key.startswith(('c3', 'c6')): continue  # C3 now has its accepted multiscale fixture suite.
            name=mapping.get(key.split('_')[0],key.split('_')[0]); seed=17
            expected,meta=func(gt,np.random.default_rng(seed)); rng=np.random.default_rng(seed)
            params={}; random_inputs={}; h,w=gt.shape[:2]
            def field():
                return torch.from_numpy(rng.normal(0,1,(h,w)).astype(np.float32))[None,None]
            if name=='motion': params['angle']=float(rng.uniform(20,160))
            if name=='c2': params['phase_xy']=rng.uniform(-.42,.42,2).astype(np.float32).tolist()
            if name=='c3': random_inputs={'cr':field(),'cb':field()}
            if name=='c4':
                f=field(); direction=rng.normal(0,1,2).astype(np.float32)
                random_inputs={'field':f,'direction':torch.from_numpy(direction)[None,:,None,None]}
            if name=='c8': random_inputs={'corners':torch.from_numpy(rng.uniform(-.008,.008,(4,2)).astype(np.float32)*np.float32([w,h]))}
            start=time.perf_counter()
            actual,record=pipeline.torch_atomic(x,name,torch.Generator().manual_seed(seed),params,random_inputs)
            elapsed=time.perf_counter()-start
            actual=actual[0].permute(1,2,0).mul(255).round().byte().numpy()[:,:,::-1]
            delta=actual.astype(float)-expected.astype(float)
            mae=float(np.abs(delta).mean()); rmse=float(np.sqrt((delta**2).mean()))
            stats_ref=statistics(gt,expected); stats_torch=statistics(gt,actual)
            # Screening bounds only; visual and C3 semantic acceptance remain separate.
            record={'atom':key,'input':index,'seed':seed,'mae_255':mae,'rmse_255':rmse,
                    'p99_abs_255':float(np.percentile(np.abs(delta),99)),
                    'cpu_seconds':elapsed,'reference_statistics':stats_ref,'torch_statistics':stats_torch,
                    'pixel_screen_pass':mae <= 2.0 and rmse <= 4.0}
            records.append(record)
            cv2.imwrite(str(out/f'{index}_{key}.png'),np.concatenate([gt,expected,actual],axis=1))
    result={'records':records,'torch_sha256':sha(ROOT/'realesrgan/lenovo_atomic_torch.py'),
            'automatic_screen_pass':all(r['pixel_screen_pass'] for r in records),
            'visual_status':'pending','gpu_throughput':'unmeasured: CPU-only environment',
            'formal_training_authorized':False}
    (out/'report.json').write_text(json.dumps(result,indent=2))
    for r in records: print(r['input'],r['atom'],round(r['mae_255'],3),round(r['rmse_255'],3),r['pixel_screen_pass'])
    print(out/'report.json')


if __name__=='__main__': main()
