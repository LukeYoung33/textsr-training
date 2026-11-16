"""IDEA-010固定mini验证集交互报告；人工评分与数值差异分离。"""
import csv
import json
from pathlib import Path


def write_report(directory, summary):
    """@luna-comment Input:报告目录/完整指标；Output:中文HTML及CSV；Key Steps:嵌入排序清单/原生图/同步变换/草稿及导出；Key Params:评价初始为空；State/Dependencies:自有报告文件。"""
    directory = Path(directory)
    columns = ['id', 'mad', 'max_abs', 'changed_fraction', 'l1_psnr', 'ori_psnr', 'deit_l1_psnr', 'deit_ori_psnr', 'b0_l1_psnr', 'b0_ori_psnr', 'v3m_psnr', 'l1_architecture_mad', 'repvit_loss_ablation_mad']
    with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in summary['rows'])
    payload = json.dumps(summary, ensure_ascii=False).replace('<', '\\u003c')
    content = r'''<!doctype html><meta charset="utf-8"><title>IDEA-010 RepViT-M0.6效果评审</title>
<style>body{font:15px sans-serif;margin:20px;background:#eef2f7;color:#172333}button,input,select{padding:8px;margin:4px} .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}canvas{width:100%;height:340px;background:#333;touch-action:none}article{background:white;padding:10px}textarea{width:95%;height:60px}#state{font-weight:bold}h2{font-size:16px}</style>
<h1>IDEA-010 · RepViT-M0.6 双损失效果评审</h1>
<p>固定48个验证图块，按RepViT R1相对同一步数DeiT R1的有效区域RGB平均绝对差降序。差异表示变化量，不代表质量。GT沿用历史583图验证集，未重建+1目标；这是裁块验证，不能替代整图测试。</p>
<p id="identity"></p><p>RepViT/DeiT/EfficientViT六组均为同一步数，损失配方分别匹配。V3M为历史R1最终500K参考，与中间里程碑预算不同。差异图为RepViT R1相对DeiT R1绝对差放大8倍。图片均256×256，可同步缩放、拖动。</p>
<button id="prev">上一张</button><select id="scene"></select><button id="next">下一张</button><button id="reset">复位</button><button id="zoom">放大2倍</button><button id="review">填写评价</button>
<p id="metrics"></p><div class="grid" id="panels"></div>
<section id="form"><h2>填写评价：RepViT R1相对同预算DeiT R1</h2><select id="score"><option value="">未评价</option><option>改善</option><option>持平</option><option>变差</option><option>不确定</option></select><input id="reviewer" placeholder="评审人"><textarea id="note" placeholder="本场景备注"></textarea><textarea id="conclusion" placeholder="总体结论"></textarea></section>
<button id="save">保存浏览器草稿</button><button id="export">导出评价JSON</button><label>导入评价JSON <input id="import" type="file" accept="application/json"></label><button id="csv">导出完整排序及评价CSV</button><button id="html">保存含评价HTML</button><p id="state"></p>
<script id="seed" type="application/json">{}</script><script>
const data=__DATA__, rows=data.rows, key='idea010-'+data.step+'-'+data.checkpoints.ori.model_sha256;
let reviews={},position=0,scale=1,dx=0,dy=0,pictures=[],drag=null;
const $=id=>document.getElementById(id), names={lq:'输入',gt:'历史GT',l1:'RepViT L1',ori:'RepViT L1-R1_ori',deit_l1:'DeiT L1',deit_ori:'DeiT L1-R1_ori',b0_l1:'EfficientViT-B0 L1',b0_ori:'EfficientViT-B0 L1-R1_ori',v3m:'V3M R1 500K参考',diff8:'RepViT R1 vs DeiT R1差异 ×8'};
try{reviews=JSON.parse(localStorage.getItem(key)||$('seed').textContent||'{}')}catch(e){$('state').textContent='草稿读取失败：'+e.message}
$('identity').textContent='RepViT步数：'+data.step+'；检查点哈希：'+data.checkpoints.ori.model_sha256;
rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' '+r.source.dataset+'/'+r.source.sample_id+' MAD '+r.mad.toFixed(4);$('scene').append(o)});
function draw(){for(const p of pictures){const c=p.canvas,ctx=c.getContext('2d');c.width=c.clientWidth;c.height=c.clientHeight;ctx.fillStyle='#333';ctx.fillRect(0,0,c.width,c.height);ctx.imageSmoothingEnabled=false;const z=Math.min(c.width/256,c.height/256)*scale;ctx.drawImage(p.image,(c.width-256*z)/2+dx,(c.height-256*z)/2+dy,256*z,256*z)}}
function draft(){const id=rows[position].id;reviews[id]={score:$('score').value,note:$('note').value};reviews.reviewer=$('reviewer').value;reviews.conclusion=$('conclusion').value;try{localStorage.setItem(key,JSON.stringify(reviews));$('state').textContent='浏览器草稿已保存（仅此浏览器）；请导出文件作持久保存'}catch(e){$('state').textContent='保存失败：'+e.message}}
function show(){
 pictures=[];$('panels').replaceChildren();$('scene').value=position;const r=rows[position];
 $('metrics').textContent='MAD '+r.mad.toFixed(5)+' / 最大差 '+r.max_abs.toFixed(3)+' / 变化比例 '+(100*r.changed_fraction).toFixed(2)+'%；PSNR RepViT L1 '+r.l1_psnr.toFixed(3)+'，RepViT R1 '+r.ori_psnr.toFixed(3)+'，DeiT L1 '+r.deit_l1_psnr.toFixed(3)+'，DeiT R1 '+r.deit_ori_psnr.toFixed(3)+'，V3M '+r.v3m_psnr.toFixed(3)+' dB；L1跨结构MAD '+r.l1_architecture_mad.toFixed(4)+'；RepViT双损失MAD '+r.repvit_loss_ablation_mad.toFixed(4);
 for(const n of Object.keys(names)){const a=document.createElement('article'),h=document.createElement('h2'),c=document.createElement('canvas'),im=new Image;h.textContent=names[n];a.append(h,c);$('panels').append(a);const p={canvas:c,image:im};pictures.push(p);im.onload=draw;im.src=r.images[n];c.onwheel=e=>{e.preventDefault();scale=Math.max(.25,Math.min(32,scale*Math.exp(-e.deltaY*.002)));draw()};c.onpointerdown=e=>{drag=[e.clientX,e.clientY,dx,dy];c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(drag){dx=drag[2]+e.clientX-drag[0];dy=drag[3]+e.clientY-drag[1];draw()}};c.onpointerup=c.onpointercancel=()=>drag=null}
 const saved=reviews[r.id]||{};$('score').value=saved.score||'';$('note').value=saved.note||'';$('reviewer').value=reviews.reviewer||'';$('conclusion').value=reviews.conclusion||'';
}
function download(name,text,type){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function navigate(i){draft();position=Math.max(0,Math.min(rows.length-1,i));show()}
$('scene').onchange=()=>navigate(+$('scene').value);$('prev').onclick=()=>navigate(position-1);$('next').onclick=()=>navigate(position+1);$('reset').onclick=()=>{scale=1;dx=dy=0;draw()};$('zoom').onclick=()=>{scale=Math.min(32,scale*2);draw()};$('review').onclick=()=>{$('form').scrollIntoView();$('score').focus()};$('save').onclick=draft;for(const id of ['score','note','reviewer','conclusion'])$(id).onchange=draft;
$('export').onclick=()=>{draft();download('IDEA010_'+data.step+'_reviews.json',JSON.stringify({key,reviews},null,2),'application/json')};
$('import').onchange=async e=>{try{const value=JSON.parse(await e.target.files[0].text());if(value.key!==key)throw Error('实验/检查点不匹配');reviews=value.reviews;localStorage.setItem(key,JSON.stringify(reviews));show();$('state').textContent='评价已导入并保存浏览器草稿'}catch(error){$('state').textContent='导入失败：'+error.message}};
$('csv').onclick=()=>{draft();const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const lines=[['id','source','mad','max_abs','changed_fraction','l1_psnr','ori_psnr','deit_l1_psnr','deit_ori_psnr','v3m_psnr','l1_architecture_mad','repvit_loss_ablation_mad','score','note'],...rows.map(r=>[r.id,r.source.dataset+'/'+r.source.sample_id,r.mad,r.max_abs,r.changed_fraction,r.l1_psnr,r.ori_psnr,r.deit_l1_psnr,r.deit_ori_psnr,r.b0_l1_psnr,r.b0_ori_psnr,r.v3m_psnr,r.l1_architecture_mad,r.repvit_loss_ablation_mad,reviews[r.id]?.score,reviews[r.id]?.note])];download('IDEA010_'+data.step+'_review.csv','\uFEFF'+lines.map(line=>line.map(quote).join(',')).join('\n'),'text/csv')};
$('html').onclick=()=>{draft();$('seed').textContent=JSON.stringify(reviews).replaceAll('<','\\u003c');const clone=document.documentElement.cloneNode(true);const base=document.createElement('base');base.href=document.baseURI;clone.prepend(base);download('IDEA010_'+data.step+'_review.html','<!doctype html>'+clone.outerHTML,'text/html')};window.onresize=draw;show();
</script>'''
    content = content.replace('__DATA__', payload)
    if summary.get('preflight_only'):
        content = content.replace('<h1>', '<p style="background:#ffe7a0;padding:15px">仅为9步CUDA预检检查点的报告链路验证，不是正式训练效果结论。</p><h1>', 1)
    (directory / 'index.html').write_text(content, encoding='utf-8')
