"""Read-only early-checkpoint native replay on the four diagnostic anchors."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_v3_edge_pilot import OUT, SOURCE, REPO, save, digest
PILOT=OUT
CONTINUATION='--continue-2k' in sys.argv
if CONTINUATION:OUT=OUT.with_name('v3_edge_cont10k_20261028')
BASE=Path('/AI_DNS/guang.yang/TextSR/data/v3pre_three_mojito280_20261028')
DIAG=BASE/'diagnostics/residual_frequency_049052_20261028'
DEST=OUT/'comparison'
ARMS={'baseline':'原 loss 续训','identity':'GT 恒等监督','guard':'可信边缘抵消保护'}


def publish(completed,records):
    """@luna-comment Input:completed checkpoint list/raw metric records. Output:Chinese native viewer and manifest. Key Steps:publish only complete model/case groups. Key Params:no image resampling. State/Dependencies:local relative assets; original template unchanged."""
    from tools.build_camera_2x2_viewer import render_html
    models=[dict(key='lq',label='V3 预处理输入',note='原生同位置区域；没有再次预处理'),dict(key='pre250',label='Full 预训练250K｜起点',note='所有实验的共同初始权重')]
    for step in completed:
        for arm,title in ARMS.items():models.append(dict(key=f'{arm}_{step}',label=f'{title}｜+{step} iter',note='同起点与采样；256/224，crop8+smooth16；未经额外锐化'))
    rows=[]
    for identity in ['049','050','051','052']:
        images={'pre250':identity+'_pre250.png'}
        for step in completed:
            for arm in ARMS:images[f'{arm}_{step}']=f'{identity}_{arm}_{step}.png'
        rows.append(dict(id=identity,name=identity+'｜原生文字区域',width=1200,height=350,lq=identity+'_input.png',images=images))
    latest=completed[-1] if completed else None
    defaults='lq,pre250,'+(f'identity_{latest},guard_{latest}' if latest else 'pre250,pre250')
    manifest=dict(title='049–052：两组强约束短训对比',subtitle='预处理输入 / 原始250K / 同程基准 / 恒等监督 / 抵消保护',models=models,rows=rows,defaults=defaults,notes='四张图是诊断锚点，不代表总体效果。先比较同一步数的实验组和原 loss 续训基准，再比较共同250K起点。保持相同原生切块和融合；loss、残差频谱及边缘投影见 METRICS.json。变锐并不自动代表改善，还应检查 halo、假笔画和噪声。')
    page=render_html(manifest).replace('← Previous','← 上一张').replace('Next →','下一张 →').replace("comparisonState='?????????'","comparisonState='加载中'")
    save(DEST/'manifest.json',manifest);save(DEST/'METRICS.json',dict(completed_steps=completed,records=records))
    temporary=DEST/'viewer.html.tmp';temporary.write_text(page,encoding='utf-8');temporary.replace(DEST/'viewer.html')


def main():
    """@luna-comment Input:saved pilot or continuation checkpoints. Output:four native ROI outputs and residual readouts per checkpoint/arm. Key Steps:retain verified2K references in continuation;wait for complete writes;hash-bound source;strict model load;original parent-grid capture;publish complete groups. Key Params:100/500/1K/2K or2K/4K/6K/10K;GPU3 only. State/Dependencies:read-only training files; isolated comparison output."""
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3pre_area import bootstrap
    bootstrap(REPO)
    from basicsr.archs import build_network
    module_spec=importlib.util.spec_from_file_location('pilot_residual_capture',DIAG/'analyze.py')
    capture=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(capture)
    cv2.setNumThreads(1);torch.set_num_threads(2);DEST.mkdir(parents=True,exist_ok=True)
    assert not (DEST/'METRICS.json').exists(),'Refusing to overwrite completed comparison'
    entries={r['id']:r for r in json.loads((BASE/'selection.json').read_text())['samples']}
    binding=json.loads((BASE/'MODEL_BINDINGS.json').read_text())['full']
    assert digest(binding['checkpoint'])==binding['checkpoint_sha256']
    from tools import restore_crop8_block256
    assert digest(restore_crop8_block256.__file__)==binding['stitcher_sha256']
    model=build_network(yaml.safe_load(Path(binding['config']).read_text())['network_g']).cuda().eval()
    for identity in capture.ROIS:
        shutil.copyfile(DIAG/(identity+'_input.png'),DEST/(identity+'_input.png'))
        shutil.copyfile(DIAG/(identity+'_full_output.png'),DEST/(identity+'_pre250.png'))
    completed=[];records=[]
    if CONTINUATION:
        previous=json.loads((PILOT/'comparison/METRICS.json').read_text())
        records=[r for r in previous['records'] if r['step']==2000]
        assert len(records)==12
        for record in records:
            filename=f'{record["id"]}_{record["arm"]}_2000.png'
            assert digest(PILOT/'comparison'/filename)==record['output_sha256']
            shutil.copyfile(PILOT/'comparison'/filename,DEST/filename)
        completed=[2000]
    publish(completed,records)
    for step in ([4000,6000,10000] if CONTINUATION else [100,500,1000,2000]):
        for arm in ARMS:
            checkpoint=OUT/arm/f'net_g_{step}.pth'
            while not (OUT/arm/f'optimizer_{step}.state').exists():
                if (OUT/arm/'FAILED.json').exists():raise RuntimeError(f'Training failed: {arm}')
                save(OUT/'EVALUATION_STATUS.json',dict(phase='waiting_checkpoint',arm=arm,step=step,completed=completed));time.sleep(10)
            model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
            checkpoint_hash=digest(checkpoint)
            for identity,display_roi in capture.ROIS.items():
                save(OUT/'EVALUATION_STATUS.json',dict(phase='inference',arm=arm,step=step,image=identity,completed=completed))
                entry=entries[identity];assert entry['orientation']==6 and digest(entry['input'])==entry['input_sha256']
                source=cv2.cvtColor(cv2.imread(entry['input'],cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
                xa,ya,xb,yb=display_roi;h=source.shape[0];roi=[ya,h-xb,yb,h-xa]
                result=capture.capture(source,model,roi)
                replay=np.rot90(result.pop('replay'),k=-1)
                assert replay.shape==(350,1200,3)
                image=DEST/f'{identity}_{arm}_{step}.png';assert cv2.imwrite(str(image),cv2.cvtColor(replay,cv2.COLOR_RGB2BGR))
                raw=DEST/f'{identity}_{arm}_{step}_residual.npz'
                np.savez_compressed(raw,x=result.pop('x'),r=result.pop('r'),center=result.pop('center'))
                result.pop('xp');result.pop('rp')
                records.append(dict(id=identity,arm=arm,step=step,checkpoint_sha256=checkpoint_hash,source_sha256=entry['input_sha256'],output_sha256=digest(image),residual_sha256=digest(raw),**result))
                print(json.dumps(dict(id=identity,arm=arm,step=step,edge_residual_projection=result['local']['gradient_residual_projection'])),flush=True)
        completed.append(step);publish(completed,records)
    save(OUT/'EVALUATION_STATUS.json',dict(phase='complete',completed=completed,records=len(records)))


if __name__=='__main__':
    try:main()
    except Exception as error:
        save(OUT/'EVALUATION_FAILED.json',dict(error=repr(error)));raise
