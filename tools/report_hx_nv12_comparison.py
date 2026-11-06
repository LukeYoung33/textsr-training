"""Publish the preprocessing-only280 column and the separate six-choice HX viewer."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.hx_nv12_comparison import OUT,DATA,sha,save
from tools.build_camera_2x2_viewer import render_html,file_url,models


def update280():
    """@luna-comment Input:accepted280 manifest/cache. Output:updated viewer plus preprocessing coverage audit. Key Steps:verify new cache hashes/shapes/source binding; keep all model URLs; inline render. Key Params:V2_2 only, no model/post in new column. State/Dependencies:existing result PNGs remain immutable."""
    from PIL import Image
    destination=DATA/'camera_compare_2x2_20261022'
    manifest=json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
    pre=DATA/'camera_robust_pre_native_v22_20261021'
    records=[]
    assert len(manifest['rows'])==280
    for row in manifest['rows']:
        receipt=pre/'receipts'/(row['id']+'.json')
        record=json.loads(receipt.read_text())
        source_receipt=json.loads((destination/'results/original_v3_pre/receipts'/(row['id']+'.json')).read_text())
        assert record['source_sha256']==source_receipt['source_sha256'] and record['outside_support_unchanged']
        image=pre/record['image']
        assert sha(image)==record['image_sha256']
        with Image.open(image) as pixels:assert pixels.size==(row['width'],row['height'])
        row['images']['pre']=file_url(image)
        records.append(dict(id=row['id'],receipt_sha256=sha(receipt),image_sha256=record['image_sha256']))
    manifest['models']=models()
    target=destination/'index.html'
    target.write_text(render_html(manifest),encoding='utf-8')
    save(destination/'manifest.json',manifest)
    save(destination/'PRE_COLUMN_AUDIT.json',dict(images=280,records=records,html_sha256=sha(target),status='passed'))
    audit=json.loads((destination/'COMPARISON_AUDIT.json').read_text())
    audit.update(html_sha256=sha(target),preprocessing_column='V2_2',preprocessing_audit='PRE_COLUMN_AUDIT.json')
    save(destination/'COMPARISON_AUDIT.json',audit)
    print('PASS updated280 preprocessing column',flush=True)


def report():
    """@luna-comment Input:61 decoded pairs and four freshly completed result columns. Output:six-choice viewer/manifest/audit. Key Steps:hash all linked finals, enforce source dimensions and exact prior model identities; inline shared UI. Key Params:device output is conversion-only, pre has no model/post. State/Dependencies:retained pipeline intermediates and immutable original dump archive."""
    from PIL import Image
    inventory=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
    accepted=json.loads((DATA/'camera_compare_2x2_20261022/COMPARISON_AUDIT.json').read_text())['bindings']
    definitions={m['key']:m for m in models()}
    choices=[dict(key='lq',label='LQ · NV12 输入转 RGB',note='input_0 原始 NV12 → RGB；不做图像增强'),dict(key='pre',label='V2_2 · 仅预处理',note='文档检测 → warp → V2_2 → unwarp贴回；不经过模型/后处理'),dict(key='device',label='设备 output · NV12 转 RGB',note='同一对 dump 中的 output；仅 NV12 解码，不再做预处理/模型/后处理')]
    choices.extend(definitions[key] for key in ('original_v3_pre','vosr_v3_pre','vosr_v5_block'))
    rows,records=[],[]
    for source in inventory['pairs']:
        identity=source['id'];asset=source['input']
        row=dict(id=identity,name=asset['name'],width=asset['width'],height=asset['height'],lq=file_url(asset['png']),images={})
        for name in ('input','output'):
            value=source[name]
            assert sha(value['png'])==value['png_sha256']
            with Image.open(value['png']) as image:assert image.size==(row['width'],row['height'])
        row['images']['device']=file_url(source['output']['png'])
        for arm in ('pre','original_v3_pre','vosr_v3_pre','vosr_v5_block'):
            receipt_file=OUT/'results'/arm/'receipts'/(identity+'.json')
            receipt=json.loads(receipt_file.read_text())
            assert receipt['source_sha256']==asset['png_sha256'] and receipt['outside_owned_unchanged']
            if arm!='pre':
                for key in ('checkpoint_sha256','config_sha256'):
                    assert receipt['binding'][key]==accepted[arm][key]
            assert receipt['binding']['preprocessing']==('none' if arm=='vosr_v5_block' else 'V2_2')
            image=OUT/receipt['final']
            assert sha(image)==receipt['final_sha256']
            with Image.open(image) as pixels:assert pixels.size==(row['width'],row['height'])
            row['images'][arm]=file_url(image)
            records.append(dict(id=identity,arm=arm,receipt_sha256=sha(receipt_file),final_sha256=receipt['final_sha256']))
        rows.append(row)
    assert len(rows)==61
    manifest=dict(title='海信 L772 · 61 对 NV12',subtitle='三条 pipeline + 设备 output\n四窗独立选择，同步缩放与切图',notes='NV12：4096×3072，stride 4096。\n解码：BT.601 limited range；input/output 使用相同转换。\n\n仅预处理：V2_2 后贴回原图。\n两组 V3：V2_2 + 模型 + 亮度后处理。\nV5 256切块：不做亮度预处理/后处理。\n\n设备 output 为对应原始 dump 直接转 RGB，未重复增强。',defaults='lq,device,original_v3_pre,vosr_v3_pre',models=choices,rows=rows)
    document_run=json.loads((OUT/'run/documents/run.json').read_text())
    no_document=[Path(v['source']).stem for v in document_run['results'] if v['instance_count']==0]
    manifest['notes']+='\n\n未检出文档：'+(', '.join(no_document) if no_document else '无')+'；按既有 pipeline 保留 LQ。'
    target=OUT/'index.html';target.write_text(render_html(manifest),encoding='utf-8')
    save(OUT/'manifest.json',manifest)
    save(OUT/'COMPARISON_AUDIT.json',dict(status='passed',images=len(rows),choices=len(choices),records=records,no_document=no_document,conversion=inventory['conversion'],html_sha256=sha(target)))
    print('PASS HX61 pairs / six choices',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('update280','report'))
    globals()[parser.parse_args().mode]()
