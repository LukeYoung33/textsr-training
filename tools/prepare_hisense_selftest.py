"""Freeze Hisense selected68 and two independently captured reference mappings."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
EXPORT=ROOT/'output/hisense_l772_selection_20261105/exports/selected_20261105_171731_3a5b57'
SOURCE=Path('Y:/guang.yang/TextSR/data/hisense_l772_two_models_20261105')
PICTURES=Path('C:/Users/Guang.Yang/Pictures/C-689_260812_海信L772文字超清')
OUT=Path('Y:/guang.yang/TextSR/data/hisense_selected_selftest_20261105')


def sha(path):
    """@luna-comment Input:file. Output:SHA256. Key Steps:stream1MiBchunks. Key Params:read-only. State/Dependencies:filesystem."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def header(path):
    """@luna-comment Input:JPEG/PNGpath. Output:storedsize/EXIForientation. Key Steps:PILheader. Key Params:no implicit alignment. State/Dependencies:read-only."""
    with Image.open(path) as image:return dict(size=list(image.size),orientation=int(image.getexif().get(274,1)))


def normalize(row,kind='original'):
    """@luna-comment Input:frozenoriginalJPEG row. Output:losslessdecodedPNG forstrictsameextensionpairing. Key Steps:decodeoriginal,PNGwrite,exactpixelroundtrip. Key Params:noresize/tonechanges. State/Dependencies:ownedOUT/original_png,sourceimmutable."""
    cv2.setNumThreads(1)
    image=cv2.imdecode(np.fromfile(row['files'][kind]['path'],np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    orientation=row['files'][kind]['orientation']
    assert orientation in (1,6)
    if orientation==6:image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    target=OUT/(kind+'_png')/(row['id']+'.png');target.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,3]);assert ok;data.tofile(target)
    assert np.array_equal(cv2.imdecode(np.fromfile(target,np.uint8),cv2.IMREAD_COLOR),image)
    return dict(id=row['id'],path=str(target),sha256=sha(target),decoded_pixels_exact=True,orientation_applied=orientation,source_sha256=sha(row['files'][kind]['path']))


def materialize_refs():
    """@luna-comment Input:frozenreferenceaudit. Output:orientation-correctnativePNG/mappings. Key Steps:explicitEXIFpixelpermutation,losslessPNG,hashreceipt. Key Params:rotation6=90CW,noresize. State/Dependencies:onlyownedreferencefolders."""
    rows=json.loads((OUT/'PAIRING_AUDIT.json').read_text(encoding='utf-8'))['rows']
    for kind in ('a55','xcover7'):
        with ThreadPoolExecutor(4) as pool:records=list(pool.map(lambda row:normalize(row,kind),rows))
        (OUT/(kind+'_normalized.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')
        (OUT/(kind+'_map.txt')).write_text('\n'.join(row['id']+'.png => '+row['id']+'.png' for row in rows)+'\n',encoding='utf-8')
        print(kind,len(records),flush=True)


def finalize():
    """@luna-comment Input:two completed68-scene reports. Output:verifieddeliveryreceipt andscopebanner. Key Steps:CSV/HTMLIDs/tileinventory checks,reference-count gate,annotateinterpretation. Key Params:no metric orverdict edits. State/Dependencies:ownedreports/localentry only."""
    import csv
    audit=json.loads((OUT/'PAIRING_AUDIT.json').read_text(encoding='utf-8'))
    for row in audit['rows']:
        assert sha(row['files']['selected']['path'])==row['selected_sha256'],(row['id'],'selected file changed')
    expected={r['id']+'.png' for r in audit['rows']}
    notice='<div style="padding:14px;background:#fff2ce;color:#332700;border:1px solid #cda63c;margin:12px 0">本报告：原始输入 → 人工选图组合（17张V3预处理、17张R1模型、34张MF/UV mix50），不是单模型统一测试。指标仅描述原始输入与所选结果的差异，不代表质量优劣。三星为独立拍摄、按方向标记旋转的视觉参考，未做几何配准，不参与指标。<br><a href="hisense_a55_sucess.html">A55参考报告</a> · <a href="hisense_xcover7_sucess.html">XCover7 Pro参考报告</a></div>'
    checks=[]
    for key in ('a55','xcover7'):
        path=OUT/f'hisense_{key}_sucess.html'
        text=path.read_text(encoding='utf-8')
        ids=set(re.findall(r'data-key="(Image\d+\.png)"',text));assert ids==expected,(key,len(ids))
        with path.with_suffix('.csv').open(encoding='utf-8-sig',newline='') as stream:csv_rows=list(csv.DictReader(stream))
        assert len(csv_rows)==68
        tiles=set(re.findall(r'data-tile="([^"]+)"',text))
        folder=path.with_name(path.stem+'_tiles')
        assert len(tiles)==204,(key,len(tiles))
        for tile in tiles:assert (folder/(tile+'.js')).stat().st_size>0
        assert text.count(chr(65533))==0
        if notice not in text:
            assert '<h1>' in text;text=text.replace('<h1>',notice+'<h1>',1);path.write_text(text,encoding='utf-8')
        checks.append(dict(reference=key,scenes=68,tiles=204,csv_rows=68,html=str(path),sha256=sha(path)))
    result=dict(passed=True,images=68,reports=checks,selection_counts=audit['selection_counts'],cross_device_metrics=False,browser_visual_test=False)
    (OUT/'DELIVERY_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    local=ROOT/'evaluation/hisense_selected_selftest_20261105'
    (local/'DELIVERY_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


def main():
    """@luna-comment Input:explicitexport/referencefolders. Output:68pairaudit/refmaps. Key Steps:exactcaptureID lookup,duplicate/missinggate,frozenselectedandinputhashes,geometry inventory. Key Params:visualrefs only. State/Dependencies:ownOUT,inputsimmutable."""
    records=json.loads((EXPORT/'selection_manifest.json').read_text())['records']
    inventory={r['id']:r for r in json.loads((SOURCE/'selection.json').read_text())['samples']}
    assert len(records)==68
    refs={}
    for key,folder in [('a55','三星A55'),('xcover7','三星xcover7 pro')]:
        refs[key]={}
        for path in (PICTURES/folder).rglob('*'):
            if not path.is_file() or path.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
            match=re.search(r'Image\d+',path.stem,re.I)
            if not match:continue
            ident=match.group().lower();assert ident not in refs[key],(key,'duplicate',ident)
            refs[key][ident]=path
    rows=[];maps={k:[] for k in refs}
    for r in records:
        src=inventory[r['source_id']]
        selected=EXPORT/'selected'/r['file']
        original=Path(src['source'].replace('/AI_DNS/','Y:/',1))
        assert sha(selected)==r['sha256'] and sha(original)==src['source_sha256']
        ident=re.search(r'Image\d+',selected.stem,re.I).group()
        matched={k:v[ident.lower()] for k,v in refs.items()}
        info={k:dict(path=str(p),**header(p)) for k,p in dict(original=original,selected=selected,**matched).items()}
        assert info['original']['size']==info['selected']['size']
        rows.append(dict(id=ident,selected_key=r['selected_key'],files=info,selected_sha256=r['sha256'],original_sha256=src['source_sha256']))
        for key,path in matched.items():maps[key].append(ident+'.png => '+path.name)
    OUT.mkdir(parents=True,exist_ok=True)
    for key,lines in maps.items():(OUT/(key+'_map.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    counts=dict(Counter(r['selected_key'] for r in records))
    audit=dict(passed=True,count=68,reference_counts={k:len(v) for k,v in refs.items()},selection_counts=counts,rows=rows)
    (OUT/'PAIRING_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    with ThreadPoolExecutor(4) as pool:normalized=list(pool.map(normalize,rows))
    (OUT/'NORMALIZED_INPUTS.json').write_text(json.dumps(normalized,indent=2),encoding='utf-8')
    print(json.dumps(dict(count=68,reference_counts=audit['reference_counts'],selection_counts=counts,
        geometry={k:dict(Counter(str(r['files'][k]['size']) for r in rows)) for k in ['original','selected','a55','xcover7']}),ensure_ascii=False))


if __name__=='__main__':
    import sys
    if '--refs-only' in sys.argv:materialize_refs()
    elif '--finalize' in sys.argv:finalize()
    else:main()
