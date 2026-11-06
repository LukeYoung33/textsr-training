"""Publish isolated local A V3 HX NV12 comparison with accepted reference columns."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NAS=Path('Y:/guang.yang/TextSR/data')
REMOTE=NAS/'hx_nv12_a_v3_vosr2u_20261026'
LOCAL=ROOT/'evaluation/hx_nv12_a_v3_vosr2u_20261026'
VIEW=ROOT/'output/hx_nv12_a_v3_vosr2u_20261026'

def copy_result(receipt):
    """@luna-comment Input: native source-bound A receipt. Output: local rehashed PNG and receipt.
    Key Steps: strict checkpoint/source geometry and byte-hash checks, copy exact PNG bytes.
    Key Params: all61 IDs, no RGB conversion or image resizing here. State/Dependencies: PIL/hash helper/NAS.
    """
    from PIL import Image
    from tools.build_vosr2u_local_three_viewer import digest
    assert receipt['binding']['checkpoint_sha256']=='ad6362d77c9895546f9c3430239cec0dc2a322d58eeed5057e4e07e0e513cc41'
    assert receipt['outside_owned_unchanged'] and receipt['source_hw']==[3072,4096]
    source=Path(receipt['final'].replace('/AI_DNS/','Y:/',1));assert digest(source)==receipt['final_sha256']
    with Image.open(source) as image:assert image.size==(4096,3072)
    for stage in receipt['stages']:
        assert stage['restoration']['canvas_hw']==[256,256] and stage['restoration']['stride_hw']==[224,224] and not stage['restoration']['ocr_cropping']
    target=LOCAL/'final'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(source,target)
    assert digest(target)==receipt['final_sha256']
    return dict(id=receipt['id'],local=str(target),receipt=receipt)

def deliver():
    """@luna-comment Input: complete61 A finals and existing upright HX manifest. Output: local native viewer/audit.
    Key Steps: exact IDs/hash-bound NV12 source correspondence, parallel checked copies, unchanged references.
    Key Params: LQ/device/oldV5/A defaults, other previous choices retained, no-document004 explicitly labeled.
    State/Dependencies: existing renderer and NAS reference images; isolated local outputs only, no quality ranking.
    """
    from tools.build_camera_2x2_viewer import render_html
    assert json.loads((REMOTE/'COMPLETE.json').read_text())['passed']
    queue=json.loads((REMOTE/'QUEUE.json').read_text());assert len(queue)==4 and all(r['exit_code']==0 for r in queue)
    source_audit=json.loads((REMOTE/'_incoming/SOURCE_AUDIT.json').read_text(encoding='utf-8'));assert source_audit['passed'] and source_audit['inputs']==61
    source_hashes={r['id']:r['sha256'] for r in source_audit['records']}
    previous=json.loads((NAS/'hx_upright_postblend_v2_20261022/manifest.json').read_text(encoding='utf-8'))
    manifest=copy.deepcopy(previous);entries=[json.loads(p.read_text()) for p in sorted((REMOTE/'receipts').glob('*.json'))]
    assert len(entries)==len(manifest['rows'])==61 and {r['id'] for r in entries}=={r['id'] for r in manifest['rows']}
    assert len(list((REMOTE/'final').glob('*.png')))==61
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(copy_result,entries))
    by_id={r['id']:r for r in records};fallback=[]
    for row in manifest['rows']:
        record=by_id[row['id']];receipt=record['receipt'];assert receipt['raw_nv12_sha256']==source_hashes[row['id']]
        row['images']['a_v3']=Path(record['local']).resolve().as_uri()
        if receipt['no_document']:fallback.append(row['id'])
    manifest['models'].append(dict(key='a_v3',label='A组 V3 · VOSR2u · 完整损失 · 250K',note='转正文档warp → 整幅256/224切块 → 融合unwarp；无亮度预处理/后处理，不用OCR框限制'))
    manifest.update(title='海信L772 NV12 · A组V3 VOSR2u',subtitle='61张输入 · LQ / 设备output / 旧V5 / A组V3',defaults='lq,device,vosr_v5_block,a_v3',notes='已逐文件核对指定camera目录的61张input_0原始NV12 SHA256，复用相同解码像素。4096×3072，stride4096，BT.601 limited range。\nA组：已完成的VOSR2u完整损失V3，250K最终权重。先转正文档warp，再256×256/步长224整幅推理，不按OCR筛文字行，无亮度预处理或亮度后处理；最后边界融合、unwarp贴回原图坐标。\n无文档检测结果的图片保留LQ：'+', '.join(fallback)+'。\nA组全图结果已下载到本地evaluation目录；LQ、设备output和旧模型引用Y盘既有结果。中间模型结果、转正warp和贴回记录在服务器独立结果目录，未改动旧页面。')
    VIEW.mkdir(parents=True,exist_ok=True)
    (VIEW/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (VIEW/'index.html').write_text(render_html(manifest),encoding='utf-8')
    (LOCAL/'SOURCE_AUDIT.json').write_text(json.dumps(source_audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOCAL/'DELIVERY.json').write_text(json.dumps(dict(passed=True,images=61,no_document=fallback,records=records),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,images=61,no_document=fallback,viewer=str(VIEW/'index.html'))))

if __name__=='__main__':deliver()
