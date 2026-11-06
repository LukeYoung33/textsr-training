"""Append matched-target evidence to the existing Chinese RCA report only."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/grid039_diagnostic_20261104'
SOURCE = Path('Y:/guang.yang/TextSR/data/grid039_diagnostic_20261104/TARGET_AUDIT.json')


def main():
    """@luna-comment Input:newmatched145patchtargetaudit. Output:additionalChinese evidence section andJSONcopy. Key Steps:requirecleanbackground/nonGTidentity,displayequalpatchaverages,markinferredcausalbridge. Key Params:no priorimagechecks ortrainingchanges. State/Dependencies:existinglocalreport/newNASaudit."""
    d = json.loads(SOURCE.read_text())
    s = d['summaries']['all']
    assert d['passed'] and d['all_non_gt_channels_equal'] and not d['mojito_included']
    assert s['dilation_raw_background_changed_patches']==0
    shutil.copy2(SOURCE,OUT/'TARGET_AUDIT.json')
    body = '<section id="training-target-evidence"><h2>更上游：共同训练样本的GT监督发生了什么变化？</h2>'
    body += '<p>从5个数据子集中分层选取20张候选，其中1张没有符合条件的实际baseline训练patch，最终采用19张、145个patch，全部不含Mojito。旧／新LQ、有效mask、几何、采样行已严格配对。背景取原始软文字mask≤12/255，向内腐蚀9px，并排除patch周边9px；145个patch中，膨胀前后这些背景像素均逐像素一致。</p>'
    body += '<p><b>证据最支持的上游候选是：替换整张GT时，也改变了非文字背景的亮度／色彩监督；两分位数的局部亮度对齐没有消除这些变化。</b>这不是“文字只粗了2px”这一种变化。</p>'
    body += '<table><tr><th>训练GT版本</th><th>对齐后背景亮度差 MAE</th><th>对齐后背景色彩差 MAE</th><th>暗背景亮度差 MAE</th><th>暗背景色彩差 MAE</th></tr>'
    for key,label in [('vosr','VOSR＋S2.2'),('original','原始GT，未增粗'),('dilate2','原始GT，增粗2px')]:
        a = s['variants'][key]['aligned']
        b = s['dark_aligned'][key]
        body += '<tr><td>'+label+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in [a['mae_y_dn'],a['mae_chroma_dn'],b['mae_y_dn'],b['mae_chroma_dn']])+'</tr>'
    body += '</table><p>单位0–255灰阶，表中是GT与LQ的监督差异，不是模型效果分数。采用patch等权平均；暗背景有44个patch，各分组与逐patch数据保存在JSON。色彩差定义为RGB残差去掉亮度分量后的绝对均值，不是ΔE。</p>'
    body += '<h3>机制链与证据边界</h3><ol><li>GT来源改变，使相同输入对应的背景／色彩目标发生变化。纯L1的前景、背景权重均为1，背景也参与优化，模型会被要求学习非恒等背景映射。</li><li>现有对比度对齐使用所有有效像素（含文字）的亮度5%和95%分位数，只对齐两个数值；对RGB三通道加相同亮度增量，截断前保留色彩分量。它不是背景逐点对齐，也不是色彩对齐。</li><li>增粗本身没有改变本次统计的远背景像素，但会改变文字占比／分位数；经过对齐后，增粗与未增粗的远背景平均RGB变化为'+f"{s['dilation_alignment_spillover_mean_dn']:.3f}"+'灰阶。这是额外的逐patch目标依赖。</li><li>下游已实测：新权重具有更强边界到内部背景响应，竖向周期条纹随切块起点移动；crop8保留了仍受影响的预测。</li></ol>'
    body += '<p><b>已证明两端：训练目标的背景／色彩差异，以及模型的边界响应／切块条纹。二者之间的训练因果链仍是最有证据支持的解释，尚未经同训练池、只换背景监督的受控训练验证。</b>共同样本不含Mojito，说明这些目标差异不依赖Mojito才存在；这不等于已通过训练消融排除移除243张的任何影响。</p>'
    body += '<p>本轮没有修改数据、loss、权重或训练。<a href="TARGET_AUDIT.json">查看完整训练目标审计</a></p></section>'
    path = OUT/'index.html'
    text = path.read_text(encoding='utf-8')
    marker = '<section id="training-target-evidence">'
    if marker in text:
        start = text.index(marker)
        end = text.index('</section>',start)+len('</section>')
        text = text[:start]+body+text[end:]
    else:
        text = text.replace('</html>',body+'</html>')
    tmp = OUT/'index.targets.tmp'
    tmp.write_text(text,encoding='utf-8')
    tmp.replace(path)
    print(json.dumps(dict(passed=True,parents=d['parents'],patches=d['patches'],background_dilation_identical=True,report=str(path))))


if __name__=='__main__':
    main()
