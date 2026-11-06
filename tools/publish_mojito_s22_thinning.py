"""Publish an isolated three-panel native Mojito morphology comparison."""
from pathlib import Path
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('Y:/guang.yang/TextSR/data')
OUT = ROOT/'evaluation/mojito280_s22_thin_20261104'


def main():
    """@luna-comment Input:existing280source manifest and completed experiment receipts. Output:local native viewer/manifest. Key Steps:bind exact IDs andfiles,adapt shared3panel renderer. Key Params:baseline/minus1/minus2,optional V3input/mask. State/Dependencies:read-only Y image paths;owned local HTML."""
    experiment = DATA/'mojito280_s22_thin_20261104'
    result = json.loads((experiment/'RESULTS.json').read_text(encoding='utf-8'))
    assert result['passed'] and len(result['rows']) == 280
    title = 'Mojito280 · VOSR S2.2 · 笔画缩窄实验'
    subtitle = '原版 / 笔画约 −1px / 笔画约 −2px'
    notes = '直接复用 VOSR + S-curve 中点斜率2.2的GT。\n单侧形态学半径0.5/1px，名义总宽度减少1/2px；斜线、交叉和细笔画处不保证固定宽度。\n黑字取局部亮值，白字取局部暗值，仅改动有效文字mask邻域。细笔画可能断开或消失，本实验不额外保护。\n不再运行OCR、模型、预处理或锐化。\n三栏同步缩放；按空格查看原版；100%查看原生像素。图片从Y盘读取。'
    models = [dict(key=k, label=label, note=note) for k, label, note in [
        ('lq', '原版 VOSR + S2.2 GT', '原生分辨率；未缩窄'),
        ('minus1', '笔画总宽度约 −1px', '单侧收缩0.5px；8方向亚像素形态学'),
        ('minus2', '笔画总宽度约 −2px', '单侧收缩1px；8方向亚像素形态学'),
        ('input', 'V3预处理输入', '对应GT生成时的输入'),
        ('mask', '文字Mask', '沿用已对齐方向的原始文字mask')]]
    unchanged = [r['id'] for r in result['rows'] if r['outputs']['minus2']['changed_pixels'] == 0]
    notes += '\n现有文字mask及OCR均为空，原样保留的编号：'+', '.join(unchanged)+'。'
    rows = []
    for row in result['rows']:
        sources = {k: Path(v.replace('/AI_DNS/guang.yang/TextSR/data', str(DATA))).as_uri() for k, v in row['source'].items()}
        images = {k: (experiment/v['path']).as_uri() for k, v in row['outputs'].items()}
        images.update(input=sources['lq'], mask=sources['text_mask'])
        rows.append(dict(id=row['id'], name=row['name'], width=row['width'], height=row['height'], lq=sources['gt'], images=images))
    data = dict(title=title, subtitle=subtitle, notes=notes, models=models, defaults='lq,minus1,minus2', rows=rows)
    template = (ROOT/'evaluation/near_val056_train558_gt_comparison_20261028/index.html').read_text(encoding='utf-8')
    template = re.sub(r'const DATA=.*?;</script>', lambda _: 'const DATA='+json.dumps(data, ensure_ascii=False)+';</script>', template, count=1, flags=re.S)
    template = re.sub(r'<title>.*?</title>', '<title>'+title+'</title>', template, count=1)
    template = re.sub(r'<aside><h1>.*?</small>', '<aside><h1>'+title+'</h1><small>'+subtitle+'</small>', template, count=1, flags=re.S)
    template = re.sub(r'<div id="notes">.*?</div>', '<div id="notes">'+html.escape(notes).replace('\n','<br>')+'</div>', template, count=1, flags=re.S)
    template = template.replace('grid-template-rows:1fr 1fr', 'grid-template-rows:1fr')
    template = template.replace('i<6;i++', 'i<3;i++')
    template = template.replace('LQ  |  模型结果', '原版 | 缩窄').replace('与原始 GT 滑动对比', '与原版滑动对比')
    template = template.replace('href="PROVENANCE.json"', 'href="RESULTS.json"')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'index.html').write_text(template, encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT/'index.html')


if __name__ == '__main__':
    main()
