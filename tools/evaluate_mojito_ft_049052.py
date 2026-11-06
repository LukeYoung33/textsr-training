"""Freeze latest three finetune checkpoints and compare four authorized samples."""
from pathlib import Path
import sys,json,shutil,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tools.evaluate_s22_guard_four_mojito280 as report
import tools.evaluate_v3pre_three_mojito280 as accepted
from tools.build_camera_2x2_viewer import render_html


def main():
    """@luna-comment Input:live three-arm checkpoints and frozen selection. Output:12 audited native predictions and viewer. Key Steps:numeric latest selection,double-hash snapshot,accepted worker,receipt validation. Key Params:049-052,crop8/smooth16,no OCR. State/Dependencies:isolated outputs,training read-only."""
    out=report.DATA/'mojito_ft_three_049052_20261030';train=report.DATA/'mojito280_guard_finetune_20261030'
    out.mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    labels={'guard':'可信边缘保护','contrast':'可信边缘保护 + 对比度对齐','contrast_nocp':'可信边缘保护 + 对比度对齐 + noCP'}
    bindings={}
    if not (out/'MODEL_BINDINGS.json').exists():
        source=json.loads((train/'CONFIG_BINDINGS.json').read_text())['runs']
        for key,label in labels.items():
            cfg=Path(source[key]['config']);assert report.digest(cfg)==source[key]['config_sha256']
            files=[p for p in (train/'experiments'/source[key]['name']/'models').glob('net_g_*.pth') if p.stem[6:].isdigit()]
            ckpt=max(files,key=lambda p:int(p.stem[6:]));sha=report.digest(ckpt)
            target=out/'models'/(key+'.pth');shutil.copy2(ckpt,target);assert report.digest(target)==sha==report.digest(ckpt)
            config=out/'models'/(key+'.yml');shutil.copy2(cfg,config)
            bindings[key]=dict(config=str(config),config_sha256=report.digest(config),checkpoint=str(target),checkpoint_sha256=sha,source_checkpoint=str(ckpt),iteration=int(ckpt.stem[6:]),label=label,stitcher_sha256=report.digest(accepted.ROOT/'tools/restore_crop8_block256.py'))
        report.save(out/'MODEL_BINDINGS.json',bindings)
        samples=json.loads((report.BASE/'selection.json').read_text())['samples'];samples=[s for s in samples if s['id'] in ['049','050','051','052']];assert len(samples)==4
        report.save(out/'selection.json',dict(samples=samples))
    bindings=json.loads((out/'MODEL_BINDINGS.json').read_text());accepted.OUT=out
    import tools.train_v3_native_regions as native
    native.ROOT=Path('/home/guang.yang/workspace/Real-ESRGAN')
    for key in labels:accepted.worker(key)
    report.OUT=out;report.LABELS=labels
    previous={r['id']:r for r in json.loads((report.BASE/'manifest.json').read_text())['rows']}
    samples=json.loads((out/'selection.json').read_text())['samples']
    checked=[report.verify(e,bindings,report.digest(out/'selection.json'),previous) for e in samples]
    models=[dict(key='lq',label='原始LQ',note='原始输入'),dict(key='v3_input',label='V3预处理输入',note='复用缓存，无重复预处理')]+[dict(key=k,label=labels[k]+f' | FT {bindings[k]["iteration"]:,} iter',note='256/224/batch8/crop8/smooth16') for k in labels]
    manifest=dict(title='Mojito 049–052 · 三组 Finetune',subtitle='固定最新checkpoint，同位置联动对比',models=models,rows=[r for r,a in checked],defaults='v3_input,guard,contrast,contrast_nocp',notes='三组实际迭代数见标签，非同迭代对比。复用V3输入；不额外做OCR、影调或对比度变换。这四张属于Mojito微调数据范围，不作为独立泛化评估。')
    report.save(out/'manifest.json',manifest);(out/'viewer.html').write_text(render_html(manifest),encoding='utf-8');report.save(out/'AUDIT.json',dict(passed=True,outputs=12,records=[a for r,a in checked]))


if __name__=='__main__':main()
