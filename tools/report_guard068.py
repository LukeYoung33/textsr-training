"""Chinese evidence-only report for the no-update068 guard measurements."""
from pathlib import Path
import sys,json,shutil,html
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=Path('Y:/guang.yang/TextSR/data');SOURCE=DATA/'guard068_diagnostic_20261103';OUT=ROOT/'evaluation/guard068_diagnostic_20261103'
LABELS={'v3m_guard':'V3M Full＋guard 500K','v3m_l1':'V3M L1-only 500K','v3m_mix50':'V3M Full无guard＋50%轻退化 500K','v5_guard':'V5 Full＋guard 500K','v5_l1':'V5 L1-only 500K','v5_mix50':'V5 Full无guard＋50%轻退化 500K','previous250':'历史V3M guard＋对齐＋noCP 250K'}


def main():
    """@luna-comment Input:actual seven-model no-update records. Output:Chinese HTML/summary/native common crops. Key Steps:copy evidence,aggregate8batches,show counterfactual andgradient limits. Key Params:068only,32patches;ROI850x260. State/Dependencies:read-onlyY,owned localreport;no quality ranking."""
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    for path in SOURCE.glob('*.json'):shutil.copyfile(path,OUT/path.name)
    data=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'));assert data['passed'] and data['no_training_updates'];summary=[]
    for key,label in LABELS.items():
        rows=[r for r in data['records'] if r['model']==key];assert len(rows)==8
        mean=lambda fn:float(np.mean([fn(r) for r in rows]))
        row=dict(key=key,label=label,native=mean(lambda r:r['native']),guard_raw=mean(lambda r:r['guard_raw']),weighted_guard=mean(lambda r:r['weighted_guard']),gt_guard=mean(lambda r:r['gt_guard_raw']),gt_guard_weighted=mean(lambda r:r['gt_guard_weighted']),active=mean(lambda r:r['guard_telemetry']['active_among_trusted']),gt_active=mean(lambda r:r['gt_guard_telemetry']['active_among_trusted']),gradient_ratio_native=mean(lambda r:r['gradients']['parameter']['guard_to_native_norm']),gradient_ratio_full=mean(lambda r:r['gradients']['parameter']['guard_to_full_norm']),cos_guard_native=float(np.mean([r['gradients']['parameter']['guard_vs_native']['cosine'] for r in rows if r['gradients']['parameter']['guard_vs_native']['cosine'] is not None])),cos_guard_full=float(np.mean([r['gradients']['parameter']['guard_vs_full_without_guard']['cosine'] for r in rows if r['gradients']['parameter']['guard_vs_full_without_guard']['cosine'] is not None])),cos_native_total=mean(lambda r:r['gradients']['parameter']['native_vs_full_plus_guard']['cosine'] or 0))
        summary.append(row)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    raw=DATA/'latest_six_500k_mojito280_20261103';sources={'input':DATA/'v3pre_three_mojito280_20261028/reference/v3_input/068.png','gt':DATA/'mojito280_reused_v3_vosr_s22_20261029/results/steep_s/068.png'}
    sources.update({k:raw/'results'/k/'final/068.png' for k in LABELS if k!='previous250'});sources['previous250']=DATA/'s22_guard_contrast_four_mojito280_20261030/results/nocp_nosdf/final/068.png'
    crops={};x,y,w,h=1000,2320,1050,320
    for key,path in sources.items():
        image=cv2.imread(str(path));assert image is not None and image.shape[:2]==(4160,3120)
        crop=image[y:y+h,x:x+w];assert cv2.imwrite(str(OUT/(key+'.png')),crop,[cv2.IMWRITE_PNG_COMPRESSION,3]);crops[key]=str(path)
    (OUT/'CROP_BINDING.json').write_text(json.dumps(dict(display_roi_xywh=[x,y,w,h],sources=crops,extra_processing=False),indent=2),encoding='utf-8')
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>068：guard为什么可能限制修复</title><style>body{font:16px/1.75 system-ui,"Microsoft YaHei";max-width:1300px;margin:30px auto;padding:0 20px;color:#18314b}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #ccd6e0;text-align:right}th:first-child,td:first-child{text-align:left}.note{padding:16px;background:#eef4fb;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{max-width:100%;border:1px solid #bbb}a{color:#006b9f}code{background:#eee;padding:2px}</style><h1>第068张：guard约束与笔画修复的冲突</h1><p class="note">本报告只诊断已保存模型；7个权重严格加载，32个实际训练bank patch，8批次，每批4张。网络和Adam均确认未更新。原始损失、门槛、有效文字mask及GT对比度对齐均复用。所有表格为8批次算术平均，不是整套Mojito280指标。</p>'
    body+='<h2>主要证据：GT也会被惩罚</h2><p>V3M当前guard固定系数为621.777。把输出直接设为该批训练GT，原生L1为0，但guard加权损失平均为'+f"{summary[0]['gt_guard_weighted']:.2f}"+'；把输出设为输入自身，guard严格为0。GT在可信支持域中约'+f"{summary[0]['gt_active']*100:.1f}%"+'的位置触发惩罚。当前guard并非以GT为最优点。</p>'
    body+='<h2>同一V3M guard下的反事实比较</h2><p>以下右列对三种预测均使用系数621.777。L1-only及混合组实际训练没有guard；右列仅说明：若给这些预测加回相同guard，会受到多大代价。</p><table><tr><th>预测</th><th>原生Y4UV L1</th><th>反事实加权guard</th><th>可信域触发比例</th></tr>'
    for row in summary[:3]:body+=f'<tr><td>{row["label"]}</td><td>{row["native"]:.6f}</td><td>{row["weighted_guard"]:.6f}</td><td>{row["active"]*100:.3f}%</td></tr>'
    body+=f'<tr><td>输出＝训练GT</td><td>0</td><td>{summary[0]["gt_guard_weighted"]:.4f}</td><td>{summary[0]["gt_active"]*100:.2f}%</td></tr></table>'
    body+='<h2>参数梯度也存在冲突</h2><p>比例＝加权guard参数梯度范数／原生L1参数梯度范数；余弦为负表示两者局部下降方向相互冲突。反事实数值不是无guard模型真实训练时的梯度。</p><table><tr><th>模型</th><th>反事实梯度范数比</th><th>guard与L1余弦</th><th>guard与Full无guard余弦</th></tr>'
    for row in summary:body+=f'<tr><td>{row["label"]}</td><td>{row["gradient_ratio_native"]:.2f}</td><td>{row["cos_guard_native"]:.3f}</td><td>{row["cos_guard_full"]:.3f}</td></tr>'
    body+='</table><p>V3M L1-only的8个批次中，假设加回相同guard，其梯度范数比均超过700，且与L1梯度余弦均为负；说明这类预测在现有guard下会受到强烈反向压力。当前guard模型自身的guard损失很小，说明它已满足约束，不代表这项约束没有限制最优解。</p>'
    proof=json.loads((OUT/'CONTROLLED_PROOF.json').read_text());body+='<h2>排除数据因素的受控反例</h2><p>构造无噪声、无错位的正弦输入；GT仅把高频幅度降到输入的70%。梯度同向、幅度比0.7，满足可信门槛。输出＝GT时，实际原函数的guard='+f"{proof['perfect_gt_guard']:.4f}"+'，原生L1=0；复制输入时guard=0。无需GT配准误差或教师模型失误，规则本身就存在冲突。</p>'
    body+='<h2>代码机制与标定</h2><p>可信门槛允许GT/输入梯度幅度比在0.5～2之间，但惩罚形式是 <code>ReLU(-a-0.1)²</code>，其中a是输出残差在输入高频方向上的局部投影。擦掉错误轮廓、白孔、振铃和粗糙纹理可能需要负投影，因此会与这个保护目标冲突。投影还经过σ=1.4的邻域平滑，保护作用会传到附近像素；门槛通过并不保证邻域内所有输入纹理都应保留。</p><p>当前V3M系数621.777来自前289 iter内仅8次非零观测，原始guard梯度小且波动；V5系数50.412，历史V3M系数107.868。这些系数不能跨结构直接当作强度排名，但固定大系数会放大离开约束可行区域时的代价。</p>'
    body+='<h2>能解释什么，尚不能分开什么</h2><ul><li>历史与当前guard模型共享输入高频保护目标，训练更久不会自动消除目标冲突，可能继续停留在相近的修复范围。</li><li>L1-only只需靠近GT，没有这项反向约束。混合组也关闭了guard，并加入GT轻退化输入；两种变化都可能有贡献。</li><li>缺少“Full无guard＋纯真实LQ”的单变量对照，当前不能定量分开辅助loss、去guard和50%退化各自的训练贡献。</li><li>本报告确认目标冲突及068上的局部梯度证据；不把32patch外推成全数据视觉排名，也不在此轮改loss或启动训练。</li></ul>'
    body+='<h2>与截图对应的整图输出局部</h2><p>下面均为同一原生坐标裁图，未加后处理或锐化。训练GT为全图S2.2结果，尚未做本报告梯度测量采用的逐patch对比度对齐。梯度统计使用真实训练256网格；这里的模型图是原256/224/crop8/smooth16整图输出，二者上下文不同。</p><div class="grid">'
    for key in ['input','gt','previous250','v3m_guard','v3m_l1','v3m_mix50','v5_guard','v5_l1','v5_mix50']:
        label={'input':'V3输入','gt':'训练GT全图（patch对比度对齐前）'}.get(key,LABELS.get(key,key));body+=f'<article><h3>{html.escape(label)}</h3><a href="{key}.png"><img src="{key}.png"></a></article>'
    body+='</div><h2>可核验记录</h2><p><a href="RESULTS.json">全部原始loss/梯度记录</a> · <a href="PATCH_BINDING.json">训练patch及来源</a> · <a href="CONTROLLED_PROOF.json">无噪声受控反例</a> · <a href="SOURCE_HASH_CHECKS.json">源文件哈希</a> · <a href="SUMMARY.json">统计摘要</a></p></html>'
    (OUT/'index.html').write_text(body,encoding='utf-8')


if __name__=='__main__':main()
