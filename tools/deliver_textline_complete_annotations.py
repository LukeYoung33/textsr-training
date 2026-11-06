"""Download corrected native text-line predictions and update existing comparison pages."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REMOTE=Path('Y:/guang.yang/TextSR/data/textline_complete_annotations_20261026')
LOCAL=ROOT/'evaluation/trainset_200samples/v3_three_models_20261026'

def copy_entry(job):
    """@luna-comment Input: suite and one source-bound receipt. Output: rehashed native local file/audit.
    Key Steps: verify model/source identity against old receipt, annotation recovery and geometry, copy bytes.
    Key Params: same historical checkpoint and original filename; outside union unchanged asserted server-side.
    State/Dependencies: Pillow/hash helper; unique owned local PNG, old outputs untouched, bounded IO pool.
    """
    from PIL import Image
    from tools.build_vosr2u_local_three_viewer import digest
    suite,entry,old=job
    assert entry['binding']==old['binding'] and entry['source_sha256']==old['source_sha256']
    assert entry['old_final_sha256']==old['final_sha256'] and entry['source_hw']==old['source_hw']
    assert entry['annotation_audit']['complete_boxes']>=entry['annotation_audit']['old_eligible_boxes']
    assert entry['annotation_audit']['policy']=='all_valid_aligned_annotations_no_training_flags'
    assert entry['restoration']['outside_unchanged'] and entry['restoration']['canvas_hw']==[96,480]
    assert entry['restoration']['boxes']==len(entry['boxes'])>=old['restoration']['boxes']
    remote=Path(entry['final'].replace('/AI_DNS/','Y:/',1))
    assert digest(remote)==entry['final_sha256']
    with Image.open(remote) as image:assert list(image.size)==entry['source_hw'][::-1]
    target=LOCAL/suite/'textline_v3_complete_annotations'/remote.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(remote,target)
    assert digest(target)==entry['final_sha256']
    return dict(id=entry['id'],local=str(target),receipt=entry)

def deliver():
    """@luna-comment Input: completed corrected783 predictions/current viewers. Output: local verified replacement.
    Key Steps: exact suite coverage and immutable source/weight correspondence, download, preserve old selection.
    Key Params: only textline current key replaced; existing block/provided groups stay identical, backups retained.
    State/Dependencies: existing renderer and mappedNAS; writes new PNG directories and current metadata atomically.
    """
    from tools.build_camera_2x2_viewer import render_html
    assert json.loads((REMOTE/'COMPLETE.json').read_text())['passed']
    queue=json.loads((REMOTE/'QUEUE.json').read_text());assert len(queue)==4 and all(r['exit_code']==0 for r in queue)
    previous=json.loads((LOCAL/'DELIVERY.json').read_text())['records']
    totals={};all_audits=[]
    for suite,total in [('train200',200),('validation',583)]:
        folder=LOCAL/suite;manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        old={r['id']:r for r in previous if '/'+suite+'/textline_v3/' in r['final']}
        entries=[json.loads(p.read_text()) for p in sorted((REMOTE/suite/'receipts').glob('*.json'))]
        assert len(entries)==len(old)==len(manifest['rows'])==total
        assert {r['id'] for r in entries}==set(old)=={r['id'] for r in manifest['rows']}
        assert len(list((REMOTE/suite/'final').glob('*.png')))==total
        with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(copy_entry,[(suite,r,old[r['id']]) for r in entries]))
        by_id={r['id']:r for r in records}
        for name in ('manifest.json','index.html'):
            backup=folder/(name+'.before_complete_annotations')
            if not backup.exists():shutil.copy2(folder/name,backup)
        for row in manifest['rows']:
            row['images'].setdefault('textline_v3_filtered',row['images']['textline_v3'])
            row['images']['textline_v3']=Path(by_id[row['id']]['local']).resolve().as_uri()
        for model in manifest['models']:
            if model['key']=='textline_v3':
                model['label']='V3 · 原文本行 · 完整标注修正版'
                model['note']='完整已对齐标注；不按训练visible/ignore筛行；原生96×480与16px融合，原权重/原输入'
        if not any(m['key']=='textline_v3_filtered' for m in manifest['models']):
            manifest['models'].append(dict(key='textline_v3_filtered',label='V3 · 文本行旧结果（标注筛选遗漏）',note='保留核对；曾按训练visible/ignore过滤，部分行未处理，不作为完整覆盖结果'))
        note='文本行已更新为完整标注修正版：沿用现成对齐坐标，不再因训练可见性标记跳过文字行；旧筛选结果可在下拉菜单中查看。其他结果组保持原文件。'
        if note not in manifest['notes']:manifest['notes']+='\n'+note
        for name,value in [('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)),('index.html',render_html(manifest))]:
            temporary=folder/(name+'.complete.tmp');temporary.write_text(value,encoding='utf-8');temporary.replace(folder/name)
        (folder/'COMPLETE_ANNOTATIONS_AUDIT.json').write_text(json.dumps(dict(passed=True,images=total,records=records),indent=2),encoding='utf-8')
        totals[suite]=dict(images=total,primary_flag_recovered_boxes=sum(r['receipt']['annotation_audit']['training_excluded_recovered'] for r in records),primary_flag_affected_images=sum(r['receipt']['annotation_audit']['training_excluded_recovered']>0 for r in records),changed_images=sum(r['receipt']['final_sha256']!=r['receipt']['old_final_sha256'] for r in records),additional_boxes=sum(len(r['receipt']['boxes'])-old[r['id']]['restoration']['boxes'] for r in records))
        all_audits+=records
    assert len(all_audits)==783
    (LOCAL/'COMPLETE_ANNOTATIONS_DELIVERY.json').write_text(json.dumps(dict(passed=True,total=783,suites=totals),indent=2),encoding='utf-8')
    print(json.dumps(totals))

if __name__=='__main__':deliver()
