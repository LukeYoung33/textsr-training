"""Extend complete S22 Mojito280 with audited prior V3M/text-line references."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote
import hashlib,json,sys
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'s22_full_four_mojito280_20261029'
OLD=DATA/'v3pre_three_mojito280_20261028'
sys.path.insert(0,str(OLD/'_build'))
KEYS={'full':'previous_full','text15':'previous_text15','adapt_l1':'previous_adapt_l1','original_v3_pre':'textline_v3_v22'}


def digest(path):
    """@luna-comment Input:read-only path. Output:SHA256. Key Steps:stream4MiB. Key Params:no mutation. State/Dependencies:NAS."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def verify(args):
    """@luna-comment Input:matching current/prior rows and audit/source. Output:verified references. Key Steps:ID/name/source/input/display equality and four full hashes. Key Params:280sources. State/Dependencies:PIL and read-only historical PNGs."""
    from PIL import Image
    row,old,audit,sample=args
    assert row['id']==old['id']==audit['id']==sample['id']
    for k in ('name','width','height','lq'):assert row[k]==old[k]
    assert row['images']['v3_input']==old['images']['v3_input']
    assert sample['source_sha256']==audit['source_sha256'] and sample['input_sha256']==audit['input_sha256']
    result={}
    for key,newkey in KEYS.items():
        url=old['images'][key];assert url.startswith('file:///Y:/')
        path=Path('/AI_DNS/'+unquote(url[len('file:///Y:/'):]))
        value=digest(path);assert value==audit['models'][key]['sha256']
        with Image.open(path) as im:assert im.size==(row['width'],row['height'])
        row['images'][newkey]=url;result[newkey]=dict(path=str(path),sha256=value)
    return row,dict(id=row['id'],outputs=result)


def main():
    """@luna-comment Input:completed280 archives. Output:ten-choice same-page viewer and separate extension audit. Key Steps:backup,full verification,accepted renderer,retain scale links,atomic replace. Key Params:no image/training mutation. State/Dependencies:owned metadata only."""
    from tools.build_camera_2x2_viewer import render_html
    for name in ('manifest.json','viewer.html'):
        p=OUT/name;backup=OUT/(p.stem+'.before_references'+p.suffix)
        if not backup.exists():backup.write_bytes(p.read_bytes())
    manifest=json.loads((OUT/'manifest.before_references.json').read_text())
    old=json.loads((OLD/'manifest.json').read_text());oldrows={r['id']:r for r in old['rows']}
    audit=json.loads((OLD/'AUDIT.json').read_text());assert audit['passed'];audits={r['id']:r for r in audit['records']}
    samples={r['id']:r for r in json.loads((OUT/'selection.json').read_text())['samples']}
    ids={f'{i:03d}' for i in range(1,281)};assert set(oldrows)==set(audits)==set(samples)=={r['id'] for r in manifest['rows']}==ids
    args=[(r,oldrows[r['id']],audits[r['id']],samples[r['id']]) for r in manifest['rows']]
    with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,args))
    lookup={m['key']:m for m in old['models']}
    for key,newkey in KEYS.items():
        m=dict(lookup[key]);m['key']=newkey
        m['label']=('Text-line V3 + V2_2 | historical96x480' if key=='original_v3_pre' else 'Previous '+m['label'])
        manifest['models'].append(m)
    manifest['rows']=[r for r,a in verified]
    manifest['title']='Mojito280 | S2.2 Full + previous V3M + text-line reference'
    manifest['subtitle']='Four current strategies | three previous V3M models | historical text-line pipeline'
    manifest['notes']+=' Previous V3M references are V3-input pretrain250K with crop8/smooth16. Text-line reference uses historical V3+V2_2,96x480 text regions and tone postprocessing;this is a full-pipeline comparison,not a same-checkpoint tiling ablation.'
    page=render_html(manifest)
    prior=(OUT/'viewer.before_references.html').read_text()
    marker='<p id="downsample-links">'
    if marker in prior:page=page.replace('<aside>','<aside>'+marker+prior.split(marker,1)[1].split('</p>',1)[0]+'</p>',1)
    result=dict(passed=True,images=280,reused_outputs=1120,choices=10,source_manifest_sha256=digest(OLD/'manifest.json'),source_audit_sha256=digest(OLD/'AUDIT.json'),original_manifest_sha256=digest(OUT/'manifest.before_references.json'),records=[a for r,a in verified])
    for name,text in [('REFERENCE_EXTENSION_AUDIT.json',json.dumps(result,indent=2)),('manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False)),('viewer.html',page)]:
        p=OUT/name;temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(text,encoding='utf-8');temp.replace(p)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
