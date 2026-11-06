"""Merge existing original/VOSR/VOSR2u V3/V5 block outputs without inference."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'camera280_six_block_comparison_20261026'
GROUPS={
 'original_v3':('six_final_camera280_samples_20261024/camera/manifest.json','original_full_v3','22c4a0c0b8aaf10f659b8669b9935d56e8f4451d72e6db6c4cf1888081f4135d'),
 'original_v5':('six_final_camera280_samples_20261024/camera/manifest.json','original_full_v5','82b63567635ddb196db2124537d2aa0bb7a25a19502c5aa8b82f0baa84c5cbe0'),
 'vosr_v3':('camera_compare_2x2_20261022/manifest.json','vosr_v3_block','ab67cf14df99117d8d16dfa06514573c44fcc23ad6ffc507a2f14caefb430d24'),
 'vosr_v5':('camera_compare_2x2_20261022/manifest.json','vosr_v5_block','3127b1ec41b97a82c19371d5435f8ec518290582dba572b6cbd191e9beb81265'),
 'vosr2u_v3':('v3_val583_abc_camera280_20261026/camera/manifest.json','a_v3','ad6362d77c9895546f9c3430239cec0dc2a322d58eeed5057e4e07e0e513cc41'),
 'vosr2u_v5':('camera_vosr2u_v5_ab_20261023/manifest.json','a_v5','e234be9383f4b93f3a296bce4a55cb7b662c7024ffd53ba5760b3294a845f90b')}

def asset_path(url):
    """@luna-comment Input: accepted mappedY image URL. Output: explicit remoteNAS path.
    Key Steps: require known prefix and decode URL. Key Params: no arbitrary roots allowed.
    State/Dependencies: existing immutable image references; no writes or image transformation.
    """
    assert url.startswith('file:///Y:/')
    return Path('/AI_DNS/'+unquote(url[len('file:///Y:/'):]))

def verify_row(job):
    """@luna-comment Input: one aligned row plus six source groups. Output: verified row/full hash audit.
    Key Steps: source/native/final hashes, expected checkpoint, same document/warp/OCR across all six arms.
    Key Params:280 original IDs, repaired017; exclude L1/textline/SR/B. State/Dependencies: PIL/read-only NAS.
    """
    from PIL import Image
    from tools.evaluate_offline_camera_matrix import sha
    identity,groups=job
    base=groups['original_v3'][identity]
    row={k:base[k] for k in ('id','name','width','height','lq')};row['images']={}
    source=asset_path(row['lq']);source_hash=sha(source)
    with Image.open(source) as image:
        orientation=image.getexif().get(274,1);stored_size=image.size
        display_size=stored_size[::-1] if orientation in (5,6,7,8) else stored_size
        assert display_size==(row['width'],row['height']),(identity,stored_size,orientation,display_size)
    audit=dict(id=identity,source_sha256=source_hash,source_stored_size=stored_size,source_exif_orientation=orientation,groups={});geometry=[]
    for key,(_,arm,checkpoint) in GROUPS.items():
        original=groups[key][identity]
        assert all(original[k]==row[k] for k in ('id','name','width','height','lq'))
        url=original['images'][arm];image_path=asset_path(url);receipt_path=image_path.parent.parent/'receipts'/(identity+'.json')
        receipt=json.loads(receipt_path.read_text())
        assert receipt['source_sha256']==source_hash and receipt['binding']['checkpoint_sha256']==checkpoint and receipt['outside_owned_unchanged']
        if 'source_hw' in receipt:assert receipt['source_hw']==[row['height'],row['width']]
        assert sha(image_path)==receipt['final_sha256']
        with Image.open(image_path) as image:assert image.size==(row['width'],row['height'])
        signature=(receipt['document_record_sha256'],[(s['warp_sha256'],s['ocr_sha256']) for s in receipt['stages']]);geometry.append(signature)
        for stage in receipt['stages']:
            detail=stage['restoration'];assert detail['canvas_hw']==[256,256] and detail['stride_hw']==[224,224] and detail['ocr_cropping'] is False
        row['images'][key]=url
        audit['groups'][key]=dict(image=str(image_path),sha256=receipt['final_sha256'],receipt=str(receipt_path),receipt_sha256=sha(receipt_path),binding=receipt['binding'])
    assert all(g==geometry[0] for g in geometry),('Mismatched document/warp/OCR provenance',identity)
    audit['document_record_sha256']=geometry[0][0]
    return row,audit

def build():
    """@luna-comment Input: six existing complete280-result manifests. Output: verified V3/V5-default viewers.
    Key Steps: exact ID sets, parallel read-only native/hash/provenance audit, shared inline renderer.
    Key Params: seven choices, two default pages; no inference, re-encoding, duplicate native files or score claims.
    State/Dependencies: existing render_html/NAS outputs; only new owned metadata/HTML writes.
    """
    from tools.evaluate_offline_camera_matrix import save_json,sha
    from tools.build_camera_2x2_viewer import render_html
    groups={};manifests={}
    for key,(relative,_,_) in GROUPS.items():
        path=DATA/relative;manifest=json.loads(path.read_text(encoding='utf-8'))
        groups[key]={r['id']:r for r in manifest['rows']};manifests[key]=dict(path=str(path),sha256=sha(path))
        assert len(manifest['rows'])==len(groups[key])==280
        assert set(groups[key])=={f'{i:03d}' for i in range(1,281)}
    with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify_row,[(i,groups) for i in sorted(groups['original_v3'])]))
    labels={'original':'原始GT','vosr':'VOSR GT','vosr2u':'VOSR2u GT · A组'}
    models=[dict(key='lq',label='LQ · 原始图像',note='同一批280张Mojito camera原图')]
    for key in GROUPS:
        family,arch=key.rsplit('_',1)
        models.append(dict(key=key,label=arch.upper()+' · '+labels[family]+' · 256切块',note='250K最终模型 / 完整损失 / 无亮度预处理后处理 / 同一文档warp / 256×256步长224'))
    manifest=dict(title='280张 · 原始GT / VOSR / VOSR2u · V3与V5切块',subtitle='六个模型可选 · 四窗同步缩放 / 切图 / 选图 / 前后翻页',notes='直接复用已经完成的六组280张结果，没有重新推理。\n六组均为256×256切块训练、250K最终权重、完整损失；VOSR2u使用A组，不包含B组、L1-only、X2SR或文本行版本。\n已逐图核对六组源图、文档记录、warp及OCR哈希一致，017均为修复版；推理采用同一256/224整幅文档warp切块，无亮度预处理/后处理，融合后贴回原图。\n默认比较三版GT训练的V3；V5预设页为同目录v5.html，也可在每个窗口下拉切换任意模型。\n原图和结果直接引用Y盘，需连接Y盘。训练批次/初始化等信息保留在审计文件中，不把本页表述为仅更换GT的严格单因素实验。',defaults='lq,original_v3,vosr_v3,vosr2u_v3',models=models,rows=[r for r,_ in verified])
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'manifest.json',manifest);(OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    v5=copy.deepcopy(manifest);v5['defaults']='lq,original_v5,vosr_v5,vosr2u_v5'
    (OUT/'v5.html').write_text(render_html(v5),encoding='utf-8')
    save_json(OUT/'AUDIT.json',dict(passed=True,images=280,model_outputs=1680,source_manifests=manifests,records=[a for _,a in verified]))
    print(json.dumps(dict(passed=True,images=280,outputs=1680,viewer=str(OUT/'index.html'))))

if __name__=='__main__':build()
