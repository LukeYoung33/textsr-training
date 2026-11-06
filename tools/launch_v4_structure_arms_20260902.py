"""Launch the V2-structure-only S1/R1/T1 V4 GT-degradation ablation arms."""
from __future__ import annotations

import copy, json, os, subprocess
from pathlib import Path
import yaml

REPO = Path('/home/guang.yang/workspace/Real-ESRGAN')
PYTHON = Path('/home/guang.yang/ViT_env/bin/python')
BASE = REPO / 'options/train_bucket_3/train_basic_v2_y4uv_v4_gtdeg_real_finetune_96x480_cp45_20260902.yml'
ROOT = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/formal/v4_structure_arms_20260902')

def make(name, port, arm):
    c = copy.deepcopy(yaml.safe_load(BASE.read_text()))
    c['name'] = name; c['manual_seed'] = 20260828
    c['path'].update({'pretrain_network_g': None, 'resume_state': None, 'strict_load_g': False,
                      'experiments_root': '/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments'})
    c['train'].update({'total_iter': 250000, 'cp_weight': 5e-4})
    c['datasets']['train'].update({'name': name + '_train', 'sources': [{'root': '/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2', 'weight': 1.0}], 'batch_size_per_gpu': 4})
    c['datasets']['val'].update({'name': name + '_val', 'sources': [{'root': '/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2', 'weight': 1.0}]})
    c['val'] = {'val_freq': 25000, 'save_img': False, 'metrics': {}}
    c['logger']['save_checkpoint_freq'] = 25000; c['dist_params']['port'] = port
    if arm == 's1': c['train'].update({'struct_hf_weight': 1.0, 'struct_hf_opt': {'boundary_weight': 5.0, 'outer_ring_radius': 3, 'outer_ring_weight': 6.0, 'stroke_weight': 1.0, 'background_weight': 0.25}})
    if arm == 'r1': c['train'].update({'recall_aware_weight_missing': 1.0, 'recall_aware_weight_false': 0.25, 'recall_aware_opt': {'ring_radius': 3, 'margin': 0.03, 'temperature': 0.01, 'min_polarity_contrast': 0.03}})
    if arm == 't1': c['train'].update({'text_gestalt_weight': 1e-3, 'text_gestalt_opt': {'type': 'MaskedTextGestaltLoss', 'recognizer_path': '/home/guang.yang/workspace/Real-ESRGAN/weights/scene_base_CRNN.pth', 'alphabet_path': '/home/guang.yang/workspace/Real-ESRGAN/realesrgan/metrics/benchmark.txt', 'feature_taps': [0,1,2], 'feature_weights': [1.4,1.4,1.4], 'attention_gain': 3.0, 'attention_dilate_radius': 1, 'attention_blur_sigma': 0.8}})
    return c

def main():
    arms = [('s1','2',make('v4_s1_structhf_v2only_250k_20260902',29601,'s1')), ('r1','3',make('v4_r1_recall_v2only_250k_20260902',29602,'r1')), ('t1','2',make('v4_t1_gestalt_v2only_250k_20260902',29603,'t1'))]
    ROOT.mkdir(parents=True, exist_ok=True); report=[]
    for label,gpu,c in arms:
        d=ROOT/label; d.mkdir(exist_ok=True); p=d/'train.yml'; p.write_text(yaml.safe_dump(c,sort_keys=False))
        with (d/'train.log').open('xb') as log:
            proc=subprocess.Popen([str(PYTHON),'realesrgan/train.py','-opt',str(p),'--launcher','none'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONUNBUFFERED':'1'},start_new_session=True)
        (d/'pid').write_text(str(proc.pid)+'\n'); report.append({'arm':label,'gpu':gpu,'pid':proc.pid,'config':str(p)})
    (ROOT/'launch_manifest.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__ == '__main__': main()
