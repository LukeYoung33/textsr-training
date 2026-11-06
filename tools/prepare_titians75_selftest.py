"""Freeze selected75 original/result/competitor correspondence for self-test CLI."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'.selftest/titians75_20261105'
OUT=Path('Y:/guang.yang/TextSR/data/titians75_selftest_20261105')
SELECTION=ROOT/'output/titians_subset75_selection_20261105'


def read(path):
    """@luna-comment Input:owned/frozen JSON path. Output:decoded metadata. Key Steps:explicit UTF8. Key Params:no legacy codepage. State/Dependencies:read-only files."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    """@luna-comment Input:source/target path. Output:SHA256. Key Steps:stream blocks. Key Params:1MiB. State/Dependencies:read-only file."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    """@luna-comment Input:owned path/value. Output:atomic UTF8 metadata. Key Steps:temporary replace. Key Params:preserve unicode. State/Dependencies:owned report staging only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def stage(job):
    """@luna-comment Input:explicit75 source/result/reference tuple. Output:copied inputs and frozen provenance. Key Steps:export hash gate,copy source/reference bytes,geometry/aspect/EXIF checks. Key Params:no reference warp or metrics. State/Dependencies:existing audited camera cache,selected export,authorized Huawei files."""
    from PIL import Image
    ident,row,selected,reference,inventory_row=job
    chosen=Path(selected['selected_file'])
    assert sha(chosen)==selected['sha256'],ident
    old=OUT/'original_input'/chosen.name
    ref=OUT/'reference'/reference.name
    source=Path(row['lq'])
    for src,dst in [(source,old),(reference,ref)]:
        expected=sha(src)
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(dst)==expected,(ident,str(dst))
    with Image.open(old) as a,Image.open(chosen) as b,Image.open(ref) as c:
        assert a.size==b.size==(row['width'],row['height']),ident
        assert c.getexif().get(274,1)==1,ident
        assert abs(a.width/a.height-c.width/c.height)<1e-6,(ident,a.size,c.size)
        ref_size=list(c.size)
    return dict(id=ident,scene=chosen.stem,original_name=row['name'],original_jpeg=inventory_row,
        original_canonical_source=str(source),original_input=str(old),original_sha256=sha(old),
        selected=str(chosen),selected_key=selected['selected_key'],selected_sha256=selected['sha256'],
        model_binding=selected.get('model_binding'),postprocessed=selected.get('postprocessed',False),
        reference_source=str(reference),reference=str(ref),reference_sha256=sha(ref),reference_size=ref_size,
        pairing='explicit user-authorized ordinal in sorted 300-image inventories',ordinal=int(ident))


def main():
    """@luna-comment Input:finished75 export plus prior300 order and Huawei300 order. Output:75 mapped triplets,reference map,provenance and exact CLI recipe. Key Steps:freeze identities,parallel stage4,compose explicit ref_map,bind old/new semantics. Key Params:full-resolution lossless,reference visual only. State/Dependencies:existing self-test skill CLI;no inference/selection changes."""
    state=read(SELECTION/'LAST_EXPORT.json');assert state['phase']=='complete' and state['completed']==75
    export=Path(state['folder']);assert read(export/'COMPLETE.json')['images']==75
    selected=read(export/'selection_manifest.json')
    assert selected['picks']==read(SELECTION/'choices.json')
    rows={r['id']:r for r in read(SELECTION/'manifest.json')['rows']}
    records={r['id']:r for r in selected['records']}
    expected=read(SELECTION/'SUBSET_AUDIT.json')['ids']
    assert set(rows)==set(records)==set(expected) and len(expected)==75
    inventory=read(ROOT/'evaluation/titians_two_500k_20261105/source_inventory.json')
    refs=read(LOCAL/'reference_inventory.json')
    assert len(inventory['files'])==len(refs['files'])==300
    assert refs['files']==[str(p) for p in sorted(Path(refs['root']).rglob('*')) if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']]
    assert len(set(refs['files']))==300
    for folder in ['original_input','reference','reports']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    jobs=[]
    for ident in expected:
        original=inventory['files'][int(ident)-1]
        assert original['name']==rows[ident]['name']==records[ident]['filename'],ident
        jobs.append((ident,rows[ident],records[ident],Path(refs['files'][int(ident)-1]),original))
    with ThreadPoolExecutor(max_workers=4) as pool:triplets=list(pool.map(stage,jobs))
    mapping='\n'.join(Path(r['selected']).name+'\t'+Path(r['reference']).name for r in triplets)+'\n'
    (LOCAL/'reference_map.tsv').write_text(mapping,encoding='utf-8')
    provenance=dict(old=dict(id='原始输入',scm=dict(describe='Lenovo Pad Titians 原始相机图（仅EXIF方向归一）'),simulate=dict(mode='external'),testset=dict(path=inventory['source'],files=75)),
        new=dict(id='用户选中结果',scm=dict(describe='逐图人工选择：V3预处理15 / GT+2px粗细R1 32 / MF+UV混合退化28'),simulate=dict(mode='external'),testset=dict(path=str(export/'selected'),files=75)),
        baseline_run=dict(scenes=75,output_dir=str(export/'selected'),params='模型均500K；选择以冻结清单为准；模型选择均含固化Y后处理。华为按用户指定的300张排序序号配对，仅视觉参考，未做图像配准；工具按画幅适配显示。'),verdicts={})
    save(LOCAL/'provenance.json',provenance)
    manifest=dict(passed=True,count=75,ids=expected,export=str(export),choices_sha256=sha(export/'selection_manifest.json'),
        source_inventory_sha256=sha(ROOT/'evaluation/titians_two_500k_20261105/source_inventory.json'),
        reference_inventory_sha256=sha(LOCAL/'reference_inventory.json'),choice_distribution=dict(collections.Counter(r['selected_key'] for r in triplets)),triplets=triplets)
    save(LOCAL/'triplets.json',manifest);save(OUT/'triplets.json',manifest)
    command=[str(Path.home()/'.codex/skills/alg-selftest-report/gen_selftest_report.py'),
        '--old',str(OUT/'original_input'),'--new',str(export/'selected'),
        '--ref',str(OUT/'reference'),'--ref_map',str(LOCAL/'reference_map.tsv'),
        '--ref_check',str(OUT/'reports/reference_pairing.html'),'--ref_label','华为 MatePad Mini',
        '--third_col','ref','--title','Titians 75张算法自测 · 原始输入 / 用户选中结果 / 华为 MatePad Mini',
        '--zones','--probes','--hist','--denoise','--notes_space','--csv','--jobs','4',
        '--provenance',str(LOCAL/'provenance.json'),'--workspace',str(ROOT/'.selftest'),
        '--output',str(OUT/'reports/titians75_selftest.html')]
    save(LOCAL/'command.json',command)
    print(json.dumps(dict(passed=True,triplets=75,choice_distribution=manifest['choice_distribution'],storage=str(OUT))))


if __name__=='__main__':main()
