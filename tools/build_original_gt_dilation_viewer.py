"""Local six-panel review using the established synchronized image viewer."""
from pathlib import Path
import json,re,html,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/original_gt_dilation30_20261102'


def main():
    """@luna-comment Input:30 completed native result receipts. Output:local six-view gallery and hardlinked selftest inputs. Key Steps:verify every image hash,embed manifest,adapt existing local viewer headings. Key Params:LQ/original/Mask/+1/+2/+3px. State/Dependencies:owned output only."""
    result=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'));assert result['passed'] and len(result['rows'])==30
    title='原始 GT 笔画膨胀 · 30 张与 LQ 对比'
    subtitle='上排：LQ / 原始对齐 GT / TextMask\n下排：笔画宽度约 +1 / +2 / +3 px'
    notes='原始 GT 来自训练库 original_gt_lineage，已对齐并经过 displayfit；未经过 VOSR 或 V3。\n四类来源分层均匀抽样：HXA 8、HXB 8、Lenovo0926 7、Lenovo1011 7。\n白字取局部亮值，黑字取局部暗值；仅扩展 TextMask 邻域且有效区域之外不改动。\n单侧半径 0.5/1/1.5 px，总笔画宽度名义增加约 1/2/3 px。对角线、交叉处可能不同。\n未进行额外配准或影调校正。用于观察笔画宽度，不能据此认定等价于相机 PSF。\n滚轮缩放、拖动联动；按空格临时切到 LQ，1:1 按钮查看原生像素。'
    models=[dict(key=k,label=label,note=note) for k,label,note in [
        ('lq','原始 LQ','对应训练配对输入；未做 V3'),('original','原始对齐 GT','VOSR/V3 之前；保留已有 displayfit'),('mask','TextMask','现成前景软 mask；不重跑 OCR'),
        ('width1','笔画约 +1 px','单侧半径 0.5 px；低强度对照'),('width2','笔画约 +2 px','单侧半径 1 px'),('width3','笔画约 +3 px','单侧半径 1.5 px；不是膨胀半径 3 px')]]
    rows=[]
    for row in result['rows']:
        for key,sha in row['output_sha256'].items():
            path=OUT/'images'/row['id']/(key+'.png');assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        rows.append({k:row[k] for k in ['id','name','width','height','lq','images']})
        for key in ['original','width3','lq']:
            folder=OUT/'report_inputs'/key;folder.mkdir(parents=True,exist_ok=True);link=folder/(row['id']+'.png')
            if not link.exists():link.hardlink_to(OUT/'images'/row['id']/(key+'.png'))
    data=dict(title=title,subtitle=subtitle,notes=notes,models=models,defaults=','.join(m['key'] for m in models),rows=rows)
    text=(ROOT/'evaluation/near_val056_train558_gt_comparison_20261028/index.html').read_text(encoding='utf-8')
    text=re.sub(r'const DATA=.*?;</script>',lambda _: 'const DATA='+json.dumps(data,ensure_ascii=False)+';</script>',text,count=1,flags=re.S)
    text=re.sub(r'<title>.*?</title>','<title>'+title+'</title>',text,count=1)
    text=re.sub(r'<aside><h1>.*?</small>','<aside><h1>'+title+'</h1><small>'+subtitle.replace('\n','<br>')+'</small>',text,count=1,flags=re.S)
    text=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+html.escape(notes).replace('\n','<br>')+'</div>',text,count=1,flags=re.S)
    text=text.replace('原始GT','LQ').replace('原始 GT参照','LQ参照')
    text=text.replace('与原始 GT 滑动对比','与 LQ 滑动对比').replace('href="PROVENANCE.json"','href="RESULTS.json"')
    (OUT/'index.html').write_text(text,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'DELIVERY_CHECKS.json').write_text(json.dumps(dict(passed=True,images_verified=180,rows=30,all_outside_support_unchanged=all(c['outside_support_unchanged'] for r in result['rows'] for c in r['checks'].values())),indent=2))


if __name__=='__main__':main()
