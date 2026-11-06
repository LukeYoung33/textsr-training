"""Review completed mask-repaired GTs through the existing native four-panel viewer."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from urllib.parse import quote
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data/vosr_s22_mask_repair_mix50_20261104')
LOCAL = ROOT/'evaluation/vosr_s22_mask_repair_mix50_20261104'


def digest(path):
    """@luna-comment Input:manifest path. Output:SHA256. Key Steps:stream immutable bytes. Key Params:4MiB blocks. State/Dependencies:read-only filesystem."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4194304), b''):
            result.update(part)
    return result.hexdigest()


def url(path):
    """@luna-comment Input:bound NAS path. Output:client mapped-Y image URL. Key Steps:require exact approved root then escape URL. Key Params:no guessed paths/copies. State/Dependencies:client Y mapping."""
    value = str(path)
    assert value.startswith('/AI_DNS/'), value
    return 'file:///Y:/'+quote(value[len('/AI_DNS/'):], safe='/')


def export_row(job):
    """@luna-comment Input:ordinal,completed row,original source row. Output:native review entry. Key Steps:bind exact per-parent receipt/original hashes;reuse original dimensions;read header only when absent. Key Params:no image resampling/reprocessing. State/Dependencies:existing immutable GT/M0/M2/LQ."""
    index, row, source = job
    path = DATA/'targets'/row['dataset']/row['sample_id']/'receipt.json'
    receipt = json.loads(path.read_text())
    assert receipt['passed'] and receipt['row']['output_sha256'] == row['output_sha256']
    assert receipt['binding']['source_hashes'] == source['output_sha256']
    old_gt = source['files']['gt']; old_mask = source['files']['text_mask']
    repaired = row['repair_mode'] == 'stroke_repair'
    if repaired:
        assert row['previous_training_target'] == dict(path=old_gt, sha256=source['output_sha256']['gt'])
        assert row['original_text_mask_lineage'] == dict(path=old_mask, sha256=source['output_sha256']['text_mask'])
    else:
        assert row['files']['gt'] == old_gt and row['files']['text_mask'] == old_mask
    size = row.get('cropped_size_wh')
    if size is None:
        from PIL import Image
        with Image.open(row['files']['gt']) as image:
            size = list(image.size)
    audit = receipt['audit']; changed = audit['changed_pixels']
    state = 'repaired' if changed else 'unchanged' if repaired else 'preserved'
    label = {'repaired':'已修复', 'unchanged':'无需改动', 'preserved':'区域mask · 保留原GT'}[state]
    return dict(id=f'{index+1:05d}', name=row['dataset']+' / '+row['sample_id']+'【'+label+'】',
                dataset=row['dataset'], repair_state=state, sample_id=row['sample_id'],
                width=int(size[0]), height=int(size[1]), lq=url(old_gt),
                images=dict(repaired=url(row['files']['gt']), mask_original=url(old_mask), mask_plus2=url(row['files']['text_mask']), input=url(row['files']['lq']), valid=url(row['files']['valid_mask'])),
                audit=dict(changed_pixels=changed, filled_pixels=audit.get('filled_pixels',0), cleaned_background_pixels=audit.get('cleaned_background_pixels',0), skipped_components=audit.get('skipped_components',0), unresolved_background_pixels=audit.get('unresolved_background_pixels',0)),
                receipt=url(path), source_gt_sha256=source['output_sha256']['gt'], repaired_gt_sha256=row['output_sha256']['gt'])


def export():
    """@luna-comment Input:completed full GT manifest and frozen source rows. Output:lightweight full review manifest. Key Steps:verify authoritative hashes;parallel receipt pairing;explicit preserved cases and counts. Key Params:8352 entries,read-only data. State/Dependencies:only new viewer metadata written."""
    complete = json.loads((DATA/'GT_COMPLETE.json').read_text())
    path = DATA/'source_view/manifests/train.jsonl'; source_path = DATA/'SOURCE_ROWS.json'
    assert complete['passed'] and complete['count'] == 8352
    assert digest(path) == complete['manifest_sha256'] and digest(source_path) == complete['source_rows_sha256']
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    source = {(r['dataset'],r['sample_id']):r for r in json.loads(source_path.read_text())}
    assert len(source) == len(rows) == complete['count']
    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(export_row, [(i,r,source[(r['dataset'],r['sample_id'])]) for i,r in enumerate(rows)]))
    assert len({r['id'] for r in entries}) == len(entries)
    manifest = dict(title='全量训练 GT · 原版 / 修复版', subtitle='原 VOSR+S2.2 · 近邻背景强诊断版\n全量8,352张 · 同步缩放与切换',
        defaults='lq,repaired,mask_original,mask_plus2',
        notes='默认上排：原GT / 修复后GT；下排：原M0 / 采用的M2。下拉框可切换V3输入与有效区域。\n8,109张执行笔画修复；Mojito243张仅有区域mask，保留原GT和原mask，右下并非+2px笔画mask。\n侧栏按数据集/处理状态筛选；左右键切换筛选后的图片。按住空格临时查看所选参照，支持框选放大、同步平移。\n完整原生图直接读取Y盘，不另行复制图片。无法可靠修复的位置保持原值；数量只作处理记录，效果由你看图判断。',
        models=[dict(key='lq',label='原 GT · VOSR+S2.2',note='本轮修复之前的实际教师GT；不是原始对齐GT'),
                dict(key='repaired',label='修复后 GT · 近邻背景强诊断版',note='实际全量处理结果；Mojito保留原GT'),
                dict(key='mask_original',label='原 TextMask · M0',note='原始文字mask；Mojito为区域mask'),
                dict(key='mask_plus2',label='采用的 TextMask · M2 / 保留原mask',note='修复来源用+2px软笔画mask；Mojito保留原区域mask'),
                dict(key='input',label='训练 LQ · V3预处理输入',note='复用现有V3预处理缓存，不重新处理'),
                dict(key='valid',label='有效区域 mask',note='原有效区域，不改变几何')],
        rows=entries, source=dict(gt_complete=str(DATA/'GT_COMPLETE.json'), manifest_sha256=complete['manifest_sha256'], datasets=dict(Counter(r['dataset'] for r in entries)), states=dict(Counter(r['repair_state'] for r in entries))))
    target=DATA/'VIEWER_MANIFEST.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')),encoding='utf-8');tmp.replace(target)
    print(json.dumps(dict(passed=True,count=len(entries),manifest=str(target),datasets=manifest['source']['datasets'],states=manifest['source']['states']),ensure_ascii=False),flush=True)


EXTRA = r'''
<script>
// @luna-comment Input:full viewer inventory. Output:dataset/status filters and selected-repair facts. Key Steps:reuse existing native panel loader;filter before navigation;display per-row audit. Key Params:no image pixel mutation or new requests beyond selected images. State/Dependencies:existing DATA/current/loadSample globals.
const filterArea=document.createElement('div');filterArea.style.cssText='display:grid;gap:6px;margin:8px 0';
filterArea.innerHTML='<select id="datasetFilter" aria-label="数据集筛选"><option value="">全部数据集</option></select><select id="repairFilter" aria-label="处理状态筛选"><option value="">全部处理状态</option><option value="repaired">已修复</option><option value="unchanged">无需改动</option><option value="preserved">保留原GT（Mojito）</option><option value="unresolved">有未修复背景的位置</option></select><small id="filterCount"></small>';
$('search').after(filterArea);
for(const [key,count] of Object.entries(DATA.source.datasets)){const o=document.createElement('option');o.value=key;o.textContent=key+' · '+count+'张';$('datasetFilter').append(o);}
const rowFacts=document.createElement('div');rowFacts.style.cssText='font-size:12px;color:#a8bad0;margin-top:5px';$('sample').after(rowFacts);
let visibleIndices=DATA.rows.map((_,i)=>i);
// @luna-comment Input:active sample. Output:literal processing facts and receipt link. Key Steps:use bound counters;label skipped/untouched regions explicitly. Key Params:no quality verdict. State/Dependencies:read-only manifest.
function showRepairFacts(){const r=DATA.rows[current],a=r.audit;rowFacts.replaceChildren();const text=document.createElement('span');text.textContent=r.repair_state==='preserved'?'此图只有区域mask：GT与mask原样保留。':'变化 '+a.changed_pixels.toLocaleString()+' px · 补齐 '+a.filled_pixels.toLocaleString()+' px · 背景修复 '+a.cleaned_background_pixels.toLocaleString()+' px · 无可靠取色的组件 '+a.skipped_components+' · 保留原值的背景 '+a.unresolved_background_pixels.toLocaleString()+' px';rowFacts.append(text);const link=document.createElement('a');link.href=r.receipt;link.target='_blank';link.textContent='　处理记录';rowFacts.append(link);for(const select of [...panels.map(p=>p.select),$('flashTarget')])for(const option of select.options)option.disabled=option.value!=='lq'&&!r.images[option.value];}
// @luna-comment Input:search/dataset/status selections. Output:matching sidebar and valid current sample. Key Steps:filter exact metadata;retain inventory IDs;move only if selection excluded. Key Params:all8352 entries available. State/Dependencies:page-only DOM.
function applyRepairFilters(){const query=$('search').value.toLowerCase(),dataset=$('datasetFilter').value,state=$('repairFilter').value;visibleIndices=[];const buttons=$('list').querySelectorAll('button');DATA.rows.forEach((r,i)=>{const pass=(!query||(r.id+' '+r.name).toLowerCase().includes(query))&&(!dataset||r.dataset===dataset)&&(!state||(state==='unresolved'?r.audit.unresolved_background_pixels>0:r.repair_state===state));buttons[i].hidden=!pass;if(pass)visibleIndices.push(i);});$('filterCount').textContent='筛选匹配 '+visibleIndices.length+' / '+DATA.rows.length+' 张';if(visibleIndices.length&&!visibleIndices.includes(current))loadSample(visibleIndices[0]);}
// @luna-comment Input:direction. Output:next visible sample. Key Steps:navigate filtered ordinal,clamp endpoints. Key Params:retains existing zoom reset semantics. State/Dependencies:shared loader.
function repairMove(direction){const at=visibleIndices.indexOf(current);if(visibleIndices.length)loadSample(visibleIndices[Math.max(0,Math.min(visibleIndices.length-1,at+direction))]);}
$('search').oninput=applyRepairFilters;$('datasetFilter').onchange=applyRepairFilters;$('repairFilter').onchange=applyRepairFilters;
$('previous').onclick=()=>repairMove(-1);$('next').onclick=()=>repairMove(1);
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||e.target.isContentEditable)return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();e.stopImmediatePropagation();repairMove(e.key==='ArrowLeft'?-1:1);}},true);
new MutationObserver(()=>{showRepairFacts();const at=visibleIndices.indexOf(current);$('previous').disabled=at<=0;$('next').disabled=at<0||at===visibleIndices.length-1;}).observe($('sample'),{childList:true});
showRepairFacts();applyRepairFilters();
</script>
'''


def render(path):
    """@luna-comment Input:exported bound inventory. Output:local full four-panel HTML plus manifest/audit. Key Steps:reuse stable renderer;add local filters/processing facts;retain source-native pixels. Key Params:no training/source/platform mutation. State/Dependencies:existing viewer template/math."""
    from tools.build_camera_2x2_viewer import render_html
    manifest=json.loads(Path(path).read_text(encoding='utf-8'));assert len(manifest['rows'])==8352
    overlay_path=LOCAL/'small_background07693_overlay.json'
    if overlay_path.exists():
        overlay=json.loads(overlay_path.read_text(encoding='utf-8'));row=next(r for r in manifest['rows'] if r['id']=='07693')
        assert row['repaired_gt_sha256']==overlay['base_repaired_sha256']
        manifest['models']=[m for m in manifest['models'] if m['key'] not in {v['key'] for v in overlay['models']}]+overlay['models']
        row['images'].update(overlay['images']);manifest['notes']+='\n可信小区域背景新版本仅用于07693；其他样本中该选项不可用，当前训练数据未替换。'
    for row in manifest['rows']:
        assert row['width']>0 and row['height']>0 and {'repaired','mask_original','mask_plus2','input','valid'}<=set(row['images'])
    text=render_html(manifest)
    text=text.replace('与 LQ','与原 GT').replace('与LQ','与原GT').replace('LQ  |  模型结果','原 GT  |  当前选项')
    original="const url=key==='lq'?row.lq:row.images[key];p.link.href=url;"
    assert text.count(original)==1
    text=text.replace(original,"const url=key==='lq'?row.lq:row.images[key];if(!url){p.status.textContent='此对比项仅在指定小样中可用，请选择其他项';p.status.className='status error';return;}p.link.href=url;")
    assert text.count('</html>')==1
    text=text.replace('</html>',EXTRA+'</html>')
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'compare.html').write_text(text,encoding='utf-8')
    (LOCAL/'comparison_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    audit=dict(passed=True,rows=len(manifest['rows']),counts=manifest['source'],input_manifest_sha256=digest(path),html_sha256=digest(LOCAL/'compare.html'),images_copied=0,processing_rerun=False,training_changed=False,platform_writes=0)
    (LOCAL/'COMPARISON_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    previous=ROOT/'evaluation/mask_gt_001_m2_direct_20261104/index.html'
    if previous.exists():
        old=previous.read_text(encoding='utf-8')
        if 'id="fullGtReviewLink"' not in old:
            anchor='<p id="fullGtReviewLink"><a href="'+(LOCAL/'compare.html').as_uri()+'">查看全量修复GT对比（8,352张）</a></p>'
            assert old.count('<div id="notes">')==1
            previous.write_text(old.replace('<div id="notes">',anchor+'<div id="notes">'),encoding='utf-8')
    print(json.dumps(dict(passed=True,rows=8352,html=str(LOCAL/'compare.html')),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');parser.add_argument('--manifest');args=parser.parse_args()
    if args.export:export()
    else:
        assert args.manifest
        render(args.manifest)
