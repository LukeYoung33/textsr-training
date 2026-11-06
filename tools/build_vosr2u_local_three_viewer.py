"""Read-only native comparison of supplied local583-image result groups."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'evaluation/vosr2u_validation'
REMOTE=Path('Y:/guang.yang/TextSR/data/v3_val583_abc_camera280_20261026')
OUT=ROOT/'output/vosr2u_local_three_comparison_20261026'


def digest(path):
    """@luna-comment
    Input: read-only image path. Output: SHA256. Key Steps: stream4MiB chunks into hash.
    Key Params: binary bytes, no decoding/re-encoding. State/Dependencies: file I/O only, no mutations.
    """
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304),b''):value.update(chunk)
    return value.hexdigest()


def build():
    """@luna-comment
    Input: three local groups/fixed583 selection and authoritative inference audit. Output: checked viewer/provenance.
    Key Steps: require complete name sets and same geometry, hash local assets, verify V3 identity before labeling.
    Key Params: third group remains separately named unknown-processing result, true LQ/GT explicit; five choices.
    State/Dependencies: PIL/immutable assets and existing renderer; writes isolated metadata/HTML only, no quality rank.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import render_html
    entries=json.loads((REMOTE/'validation/selection.json').read_text(encoding='utf-8'))['samples']
    reference=json.loads((REMOTE/'validation/manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((REMOTE/'validation/AUDIT.json').read_text(encoding='utf-8'))
    expected={r['id']:r['finals'] for r in audit['records']}
    assert len(entries)==len(reference['rows'])==583
    definitions=dict(provided_full='LQ_full_250000',from_scratch='V3_from-scratch',l1_only='V3_L1-only')
    for key,name in definitions.items():
        files={p.stem for p in (SOURCE/name).glob('*.png')}
        wanted={e['row']['sample_id'] if key=='provided_full' else e['id'] for e in entries}
        assert files==wanted,(name,len(files),len(wanted),sorted(files^wanted)[:10])
    rows=[];records=[];matches=dict(from_scratch=0,l1_only=0)
    for entry,base in zip(entries,reference['rows']):
        assert entry['id']==base['id']
        item={k:base[k] for k in ('id','name','width','height','lq')}
        item['images']=dict(gt=base['images']['gt']);record=dict(id=entry['id'],sample_id=entry['row']['sample_id'],groups={})
        for key,name in definitions.items():
            path=SOURCE/name/((entry['row']['sample_id'] if key=='provided_full' else entry['id'])+'.png')
            with Image.open(path) as image:assert image.size==(item['width'],item['height']),(str(path),image.size)
            stat=path.stat();sha=digest(path)
            record['groups'][key]=dict(path=str(path),sha256=sha,size=stat.st_size,mtime_ns=stat.st_mtime_ns)
            item['images'][key]=path.resolve().as_uri()
            if key!='provided_full':
                model='a_v3' if key=='from_scratch' else 'l1_v3'
                matches[key]+=int(sha==expected[entry['id']][model])
        rows.append(item);records.append(record)
    assert matches==dict(from_scratch=583,l1_only=583),matches
    models=[dict(key='lq',label='真实 LQ · 原始验证输入',note='权威验证清单中的原始LQ；与LQ_full_250000结果组区分'),dict(key='provided_full',label='LQ_full_250000 · 提供结果',note='保留目录原名；不是相同像素的原始LQ，具体处理策略未确认'),dict(key='from_scratch',label='V3_from-scratch · A组完整损失',note='583/583 SHA256匹配本轮A组V3完整损失结果'),dict(key='l1_only',label='V3_L1-only · L1-only',note='583/583 SHA256匹配本轮V3 L1-only结果'),dict(key='gt',label='GT · 旧版GT_VOSR验证目标',note='固定583对旧验证目标，仅供参照')]
    manifest=dict(title='本地三组验证结果 · 583张',subtitle='三组结果＋真实LQ / GT · 同步缩放与切图',notes='三个本地目录均完整583张，按固定验证清单对应原始文件名与001～583编号；全量尺寸已核对。\n两个V3目录逐图哈希与本轮完整损失/L1-only结果一致。\nLQ_full_250000为独立提供结果，首张与原始LQ、GT像素均不同；不推断它的模型身份。\n真实LQ和GT保留Y盘引用，需连接Y盘；原目录和ZIP未改动，不重复推理或重新编码。仅提供看图，不判定哪组更好。',defaults='lq,provided_full,from_scratch,l1_only',models=models,rows=rows)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    evidence=ROOT/'evaluation/vosr2u_local_three_comparison_20261026';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'ASSET_AUDIT.json').write_text(json.dumps(dict(passed=True,rows=583,local_assets=1749,model_matches=matches,records=records),indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=583,model_matches=matches,viewer=str(OUT/'index.html'))))


if __name__=='__main__':build()
