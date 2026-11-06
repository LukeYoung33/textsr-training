"""Frozen V5 pretrain/finetune and V3M stopped-control camera evaluation."""
from pathlib import Path
import sys,json,shutil,os
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_v3pre_training import save,sha as digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'titians_camera_three_20261030'
SOURCE=DATA/'T-026_261029_联想文字超分售前/T-026_261031_联想文字超分售前/联想Pad_Titians/camera'


def prepare(item):
    """@luna-comment Input:enumerated source. Output:canonical original and V3 cache. Key Steps:EXIF transpose once;accepted V3 estimator/repair/render;hash receipts. Key Params:strength1,chunk128,no OCR. State/Dependencies:read-only camera source."""
    import cv2,numpy as np
    from PIL import Image,ImageOps
    from evaluation.document_box_neighbor_20261027 import neighbor
    from evaluation.document_box_pre640_revision2_20261027 import revision2
    from evaluation.document_box_pre640_astra_20261027 import pre640
    cv2.setNumThreads(1);number,src=item;ident=f'{number:03d}';folder=OUT/'inputs'/ident;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json';sha=digest(src)
    if receipt.exists():
        row=json.loads(receipt.read_text());assert row['source_sha256']==sha and digest(row['input'])==row['input_sha256'];return row
    with Image.open(src) as im:image=cv2.cvtColor(np.asarray(ImageOps.exif_transpose(im).convert('RGB')),cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(folder/'lq.png'),image)
    estimate=revision2.estimate(image);q,tail=neighbor.descriptors(estimate['canvas'],estimate['nodes'])
    low,_=neighbor.repair(estimate['fields']['low'],estimate['fields']['high'],q,tail,estimate['nodes'])
    packed=dict(nodes=estimate['nodes'],tables=neighbor.tables(low,estimate['fields']['high']),native_shape=list(image.shape[:2]))
    output,_=pre640.render(image,packed,strength=1.,chunk=128);target=folder/'v3.png';assert cv2.imwrite(str(target),output)
    row=dict(id=ident,name=src.name,input=str(target),input_sha256=digest(target),source_sha256=sha,source=str(src),stored_shape=list(image.shape),height=image.shape[0],width=image.shape[1],orientation=1,split='external_camera',sample_id=src.stem,lq=str(folder/'lq.png'))
    save(receipt,row);return row


def main():
    """@luna-comment Input:authorized160 camera files and latest saved models. Output:480 audited predictions and shared viewer. Key Steps:freeze weights before preprocessing;reuse accepted native worker;validate hashes and geometry. Key Params:crop8/smooth16,batch8,no test-time contrast matching. State/Dependencies:isolated run,training untouched."""
    import yaml
    import tools.evaluate_v3pre_three_mojito280 as accepted
    from tools.build_camera_2x2_viewer import render_html,file_url
    OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    sources={'v5_pre':DATA/'vosr_2u_block256_v1_20261023/training_comparison_20261023/configs/05_v5_block256_raw.yml','v5_ft':DATA/'mojito280_v5_guard_ft_20261030/config.yml','v3m_ft':DATA/'mojito280_guard_finetune_20261030/configs/contrast_nocp.yml'}
    labels={'v5_pre':'V5 Pretrain','v5_ft':'V5 Finetune guard + 对比度对齐 + noCP','v3m_ft':'V3M Finetune guard + 对比度对齐 + noCP | LR 1e-5'}
    if not (OUT/'MODEL_BINDINGS.json').exists():
        bindings={}
        for key,config in sources.items():
            cfg=yaml.safe_load(config.read_text());models=Path(cfg['path']['experiments_root'])/cfg['name']/'models'
            checkpoint=max([p for p in models.glob('net_g_*.pth') if p.stem[6:].isdigit()],key=lambda p:int(p.stem[6:]));sha=digest(checkpoint)
            target=OUT/'models'/(key+'.pth');shutil.copy2(checkpoint,target);assert sha==digest(target)==digest(checkpoint)
            conf=OUT/'models'/(key+'.yml');shutil.copy2(config,conf)
            bindings[key]=dict(config=str(conf),config_sha256=digest(conf),checkpoint=str(target),checkpoint_sha256=sha,source_checkpoint=str(checkpoint),iteration=int(checkpoint.stem[6:]),stitcher_sha256=digest(accepted.ROOT/'tools/restore_crop8_block256.py'))
        save(OUT/'MODEL_BINDINGS.json',bindings)
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    files=sorted(p for p in SOURCE.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp'] and p.is_file());assert len(files)==160
    from tools.prepare_v3pre_training import REFERENCES
    assert all(digest(ROOT/k)==v for k,v in REFERENCES.items())
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(prepare,enumerate(files,1)):
            rows.append(row);save(OUT/'STATUS.json',dict(phase='V3 preprocessing',completed=len(rows),total=160))
    save(OUT/'selection.json',dict(samples=rows));accepted.OUT=OUT
    import tools.train_v3_native_regions as native
    native.ROOT=Path('/home/guang.yang/workspace/Real-ESRGAN')
    for key in sources:
        save(OUT/'STATUS.json',dict(phase='inference',arm=key));accepted.worker(key)
    displayed=[]
    for row in rows:
        images={'v3_input':file_url(row['input'])}
        for key in sources:
            p=OUT/'results'/key/'final'/(row['id']+'.png');r=json.loads((OUT/'results'/key/'receipts'/(row['id']+'.json')).read_text())
            assert r['binding']==bindings[key] and r['input_sha256']==row['input_sha256'] and digest(p)==r['final_sha256'];images[key]=file_url(p)
        displayed.append(dict(id=row['id'],name=row['name'],width=row['width'],height=row['height'],lq=file_url(row['lq']),images=images))
    models=[dict(key='lq',label='原始LQ',note='EXIF方向归一化'),dict(key='v3_input',label='V3预处理输入',note='全图V3，不做OCR')]+[dict(key=k,label=labels[k]+f' | {bindings[k]["iteration"]:,} iter',note='native256/224/batch8/crop8/smooth16') for k in sources]
    manifest=dict(title='Lenovo Pad Titians camera · 三模型对比',subtitle='160张 · 固定最新已保存权重',models=models,rows=displayed,defaults='v3_input,v5_pre,v5_ft,v3m_ft',notes='三模型均使用相同V3输入。V3M运行到约145K但使用实际最新保存权重，迭代数见标签。无额外OCR、warp或测试时对比度匹配。')
    save(OUT/'manifest.json',manifest);(OUT/'index.html').write_text(render_html(manifest),encoding='utf-8');save(OUT/'COMPLETE.json',dict(passed=True,images=160,outputs=480));save(OUT/'STATUS.json',dict(phase='complete',outputs=480))


if __name__=='__main__':main()
