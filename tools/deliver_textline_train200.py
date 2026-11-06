"""Verify/download native predictions and publish local synchronized viewers."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REMOTE=Path('Y:/guang.yang/TextSR/data/textline_val583_train200_20261026')
OUT=ROOT/'evaluation/trainset_200samples/v3_three_models_20261026'

def deliver():
    """@luna-comment Input: completed frozen583+600 inference. Output: verified local nativePNG/viewers.
    Key Steps: exact receipts/counts/hash/geometry, copy and rehash, bind unchanged source JPEGs and old val arms.
    Key Params: original filenames; five validation/four train choices; no re-encoding or quality metrics.
    State/Dependencies: PIL/NAS/inline renderer; only new dated directory writes, previous inputs immutable.
    """
    from PIL import Image
    from tools.build_vosr2u_local_three_viewer import digest
    from tools.build_camera_2x2_viewer import render_html
    queue=json.loads((REMOTE/'QUEUE.json').read_text());assert len(queue)==8 and all(x['exit_code']==0 for x in queue)
    bindings=json.loads((REMOTE/'MODEL_BINDINGS.json').read_text())
    assert bindings['a_v3']['checkpoint_sha256']=='ad6362d77c9895546f9c3430239cec0dc2a322d58eeed5057e4e07e0e513cc41'
    assert bindings['l1_v3']['checkpoint_sha256']=='7f09e8cf361524b99a8d4ff978a542aa17a74cbde453c9a1aa3e112420c3c16c'
    old=Path('Y:/guang.yang/TextSR/data/v3_val583_abc_camera280_20261026/validation')
    reference=json.loads((old/'manifest.json').read_text(encoding='utf-8'))
    old_audit={r['id']:r['finals'] for r in json.loads((old/'AUDIT.json').read_text())['records']}
    labels=dict(textline_v3='V3 · 原来的文本行版本',a_v3='V3 · VOSR2u 切块 · 完整损失',l1_v3='V3 · VOSR2u 切块 · L1-only')
    audit=[]
    for suite,arms,total in [('train200',tuple(labels),200),('validation',('textline_v3',),583)]:
        entries=json.loads((REMOTE/suite/'selection.json').read_text())['samples'];assert len(entries)==total
        rows=[]
        for item in entries:
            source=Path(item['input'].replace('/AI_DNS/','Y:/',1))
            if suite=='train200':source=ROOT/'evaluation/trainset_200samples/train_LQ_200/LQ_200'/source.name
            assert digest(source)==item['source_sha256']
            with Image.open(source) as image:w,h=image.size
            if suite=='validation':
                base=reference['rows'][int(item['id'])-1];assert base['id']==item['id']
                row={k:base[k] for k in ('id','name','width','height','lq')};row['images']=dict(base['images'])
                assert [base['width'],base['height']]==[w,h]
                for arm in ('a_v3','l1_v3'):
                    previous=old/arm/'final'/(item['id']+'.png')
                    assert digest(previous)==old_audit[item['id']][arm]
                    with Image.open(previous) as image:assert image.size==(w,h)
            else:row=dict(id=item['id'],name=source.stem,width=w,height=h,lq=source.resolve().as_uri(),images={})
            for arm in arms:
                receipt=json.loads((REMOTE/suite/arm/'receipts'/(item['id']+'.json')).read_text())
                assert receipt['source_sha256']==item['source_sha256'] and receipt['binding']==bindings[arm]
                assert receipt['source_hw']==[h,w]
                if arm=='textline_v3':
                    assert receipt['restoration']['outside_unchanged'] and receipt['restoration']['canvas_hw']==[96,480]
                remote=Path(receipt['final'].replace('/AI_DNS/','Y:/',1));assert digest(remote)==receipt['final_sha256']
                with Image.open(remote) as image:assert image.size==(w,h)
                target=OUT/suite/arm/remote.name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(remote,target)
                assert digest(target)==receipt['final_sha256']
                row['images'][arm]=target.resolve().as_uri()
                audit.append(receipt)
            rows.append(row)
        for arm in arms:
            assert len(list((REMOTE/suite/arm/'final').glob('*.png')))==total
            assert len(list((REMOTE/suite/arm/'receipts').glob('*.json')))==total
            assert len(list((OUT/suite/arm).glob('*.png')))==total
        models=[dict(key='lq',label='LQ · 原始输入',note='200张使用提供的实际JPG；验证集使用固定原始LQ')]
        models += [dict(key=k,label=v,note='250K最终权重；无亮度预处理/后处理；文本行原生96×480，切块256×256/步长224') for k,v in labels.items()]
        if suite=='validation':models.append(dict(key='gt',label='GT · 旧版验证目标',note='固定583对验证目标；不作为推理输入'))
        manifest=dict(title='V3三组 · '+('训练输入200张' if suite=='train200' else '完整验证集583张'),subtitle='LQ / 原文本行 / VOSR2u完整损失 / VOSR2u L1-only',notes='所有新结果已下载到本页所在目录，原始文件名保留。\n文本行沿用原生96×480、OCR区域贴回与16px融合；两组切块沿用256/224整图推理。无文档warp及亮度预处理/后处理。\n验证集完整损失和L1结果复用此前已完成结果（Y盘）；验证LQ/GT也需Y盘。训练200张四组图均为本地文件。\n历史文本行与新切块版训练批次和策略不同；此页展示效果，不作为单因素消融结论。',defaults='lq,textline_v3,a_v3,l1_v3',models=models,rows=rows)
        folder=OUT/suite;folder.mkdir(parents=True,exist_ok=True)
        (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/'index.html').write_text(render_html(manifest),encoding='utf-8')
    (OUT/'MODEL_BINDINGS.json').write_text(json.dumps(bindings,indent=2),encoding='utf-8')
    (OUT/'DELIVERY.json').write_text(json.dumps(dict(passed=True,new_finals=len(audit),records=audit),indent=2),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>V3模型对比</h1><p><a href="train200/index.html">训练输入200张：LQ＋三组V3</a></p><p><a href="validation/index.html">完整验证583张：原文本行＋此前完整损失/L1</a></p>',encoding='utf-8')
    print(json.dumps(dict(passed=True,local_finals=len(audit),output=str(OUT))))

def add_provided():
    """@luna-comment Input: supplied200 PNG results and existing train viewer. Output: fifth selectable group.
    Key Steps: exact stem-set/native-size checks and byte hashes, update manifest and inline HTML.
    Key Params: retain defaults and literal result label; no inferred model identity or image conversion.
    State/Dependencies: PIL/existing renderer; local metadata only, source images remain immutable.
    """
    from PIL import Image
    from tools.build_vosr2u_local_three_viewer import digest
    from tools.build_camera_2x2_viewer import render_html
    source=ROOT/'evaluation/trainset_200samples/train_LQ_200_full_250000/LQ_200_full_250000'
    folder=OUT/'train200';manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    files={p.stem:p for p in source.glob('*.png')}
    assert len(manifest['rows'])==len(files)==200
    assert set(files)=={r['name'] for r in manifest['rows']}
    records=[]
    for row in manifest['rows']:
        path=files[row['name']]
        with Image.open(path) as image:assert image.size==(row['width'],row['height']),(path.name,image.size)
        records.append(dict(id=row['id'],path=str(path),sha256=digest(path)))
        row['images']['provided_full']=path.resolve().as_uri()
    manifest['models']=[m for m in manifest['models'] if m['key']!='provided_full']
    manifest['models'].append(dict(key='provided_full',label='LQ_200_full_250000 · 提供结果',note='直接引用提供的对应PNG；保留目录名，未推断其模型身份'))
    manifest['title']='训练200张 · 五组可选对比'
    manifest['subtitle']='LQ / 原文本行 / VOSR2u完整损失 / VOSR2u L1-only / LQ_200_full_250000'
    note='新增 LQ_200_full_250000：200张按原文件名和尺寸核对，直接使用本地PNG，可在任意窗口下拉选择。'
    if note not in manifest['notes']:manifest['notes']+='\n'+note
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'index.html').write_text(render_html(manifest),encoding='utf-8')
    (folder/'PROVIDED_FULL_AUDIT.json').write_text(json.dumps(dict(passed=True,images=200,records=records),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,added=200,choices=len(manifest['models']))))

if __name__=='__main__':
    if '--add-provided-only' not in sys.argv:deliver()
    add_provided()
