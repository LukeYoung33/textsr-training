"""Show every matched training parent and patch behind the G039 target audit."""
from pathlib import Path
import argparse
import html
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
from tools.build_camera_2x2_viewer import file_url

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'grid039_target_candidates_20261104'


def extract():
    """@luna-comment Input:frozen19parent145patchtargetaudit/banks. Output:1160displayPNGs,parentthumbnails andviewer inventories. Key Steps:exactrowreplay,source/raw/alignedvariants,backgroundmetricgates,fullparentROIboxes. Key Params:allcandidates,noselectionbydifference;8bitdisplayonly. State/Dependencies:immutableGT/LQ/masks;CPUonly,noOCR/inference."""
    import cv2
    import numpy as np
    from tools.patch_contrast_alignment import match_patch_contrast
    from tools.diagnose_grid039_targets import measure
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    audit = json.loads((DATA/'grid039_diagnostic_20261104/TARGET_AUDIT.json').read_text())
    assert audit['passed'] and audit['summaries']['all']['dilation_raw_background_changed_patches']==0
    oldbank = DATA/'latest_vosr_four_500k_20261102/bank'
    newbank = DATA/'v3m_original_dilate2_l1_500k_20261103/bank'
    lookup = {}
    for label,bank in [('old',oldbank),('new',newbank)]:
        lookup[label] = {(s['dataset'],s['sample_id']):s for s in json.loads((bank/'train.json').read_text())['shards']}
    jobs = {(j['dataset'],j['sample_id']):j for j in json.loads((DATA/'original_aligned_gt_dilate2_20261103/SOURCE_PLAN.json').read_text())['jobs']}
    dilated = {}
    for line in (DATA/'original_aligned_gt_dilate2_20261103/manifests/train.jsonl').read_text().splitlines():
        r = json.loads(line)
        dilated[r['dataset'],r['sample_id']] = r['files']['gt']
    grouped = {}
    for index,r in enumerate(audit['records'],1):
        grouped.setdefault((r['dataset'],r['sample_id']),[]).append((index,r))
    rows = []
    parents = []
    checks = []
    for parentindex,(identity,records) in enumerate(grouped.items(),1):
        j = jobs[identity]
        full_lq = cv2.imread(j['files']['lq'])
        original = cv2.imread(j['files']['gt'])
        softmask = cv2.imread(j['files']['text_mask'],0)
        assert full_lq.shape==original.shape and softmask.shape==original.shape[:2]
        h,w = original.shape[:2]
        old = lookup['old'][identity]
        new = lookup['new'][identity]
        a = np.load(newbank/new['array'],mmap_mode='r')
        b = np.load(oldbank/old['array'],mmap_mode='r')
        scale = min(840/w,600/h,1)
        thumbnail = cv2.resize(full_lq,(round(w*scale),round(h*scale)))
        areas = []
        for index,record in records:
            key = f'P{index:03d}'
            rowid = record['row']
            x,y = record['xy']
            av,bv = np.array(a[rowid]),np.array(b[rowid])
            assert np.array_equal(av[...,:3],bv[...,:3]) and np.array_equal(av[...,6:],bv[...,6:])
            raw = dict(lq=av[...,:3],vosr_raw=bv[...,3:6],original_raw=original[y:y+256,x:x+256],dilate2_raw=av[...,3:6])
            valid = (av[...,7].astype(np.float32)/255)[None]
            support = ((softmask[y:y+256,x:x+256]<=12)&(av[...,7]>=254)).astype(np.uint8)
            support = cv2.erode(support,np.ones((19,19),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            assert int(support.sum())==record['background_pixels']
            assert np.array_equal(raw['original_raw'][support],raw['dilate2_raw'][support])
            lq = raw['lq'][...,::-1].transpose(2,0,1).astype(np.float32)/255
            values = dict(raw)
            for label in ['vosr','original','dilate2']:
                gt = raw[label+'_raw'][...,::-1].transpose(2,0,1).astype(np.float32)/255
                adjusted,stats = match_patch_contrast(gt,lq,valid)
                measured = measure(adjusted,lq,support)
                for metric,value in measured.items():
                    assert abs(value-record['variants'][label]['aligned'][metric])<1e-4,(key,label,metric)
                values[label+'_aligned'] = np.clip(np.rint(adjusted.transpose(1,2,0)[...,::-1]*255),0,255).astype(np.uint8)
            values['background_mask'] = np.repeat((support.astype(np.uint8)*255)[...,None],3,axis=2)
            folder = OUT/'patches'/key
            folder.mkdir(parents=True,exist_ok=True)
            paths = {}
            hashes = {}
            for label,value in values.items():
                p = folder/(label+'.png')
                assert value.shape==(256,256,3) and cv2.imwrite(str(p),value,[cv2.IMWRITE_PNG_COMPRESSION,3])
                paths[label] = file_url(p)
                hashes[label] = digest(p)
            name = identity[0]+' / '+identity[1]+f' · row {rowid} · ({x},{y})'
            rows.append(dict(id=key,name=name,width=256,height=256,lq=paths.pop('lq'),images=paths,
                             parent=f'T{parentindex:02d}',metrics=record))
            checks.append(dict(id=key,source_row=rowid,parent=list(identity),output_sha256=hashes,aligned_metrics_replayed=True))
            x0,y0,x1,y1 = [round(v*scale) for v in [x,y,x+256,y+256]]
            cv2.rectangle(thumbnail,(x0,y0),(x1,y1),(0,180,255),2)
            cv2.putText(thumbnail,key,(x0+2,min(y0+17,thumbnail.shape[0]-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(thumbnail,key,(x0+2,min(y0+17,thumbnail.shape[0]-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            areas.append(dict(id=key,coords=[x0,y0,x1,y1]))
        thumb = OUT/'thumbnails'/f'T{parentindex:02d}.jpg'
        thumb.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(thumb),thumbnail)
        parents.append(dict(id=f'T{parentindex:02d}',name=identity[0]+' / '+identity[1],width=w,height=h,
            lq=file_url(j['files']['lq']),images=dict(vosr_raw=file_url(j['previous_target']['path']),
            original_raw=file_url(j['files']['gt']),dilate2_raw=file_url(dilated[identity])),
            thumbnail='thumbnails/'+thumb.name,areas=areas,patches=[f'P{i:03d}' for i,r in records]))
        save(OUT/'PROGRESS.json',dict(parents_done=parentindex,total_parents=19,patches_done=len(rows),total_patches=145))
        print('PARENT',parentindex,'PATCHES',len(rows),flush=True)
    assert len(rows)==145 and len(parents)==19 and len(checks)==145
    save(OUT/'PATCHES.json',dict(rows=rows))
    save(OUT/'PARENTS.json',dict(rows=parents))
    save(OUT/'AUDIT.json',dict(passed=True,parents=19,patches=145,pngs=1160,source_audit_sha256=digest(DATA/'grid039_diagnostic_20261104/TARGET_AUDIT.json'),records=checks))
    save(OUT/'COMPLETE.json',dict(passed=True,parents=19,patches=145,pngs=1160))


def publish():
    """@luna-comment Input:audited19/145newvisualization inventory. Output:localoverview/full4view/patch8view withstatistics. Key Steps:reuseacceptedrenderer,linkallcandidates,copyonly19thumbnails/JSON. Key Params:PNGtargets8bitdisplay,trainstatisticsfloat. State/Dependencies:read-onlyYassets;noexistingmodelpagechanges."""
    from tools.build_camera_2x2_viewer import render_html
    source = Path('Y:/guang.yang/TextSR/data/grid039_target_candidates_20261104')
    destination = ROOT/'evaluation/grid039_target_candidates_20261104'
    assert json.loads((source/'COMPLETE.json').read_text())['passed']
    destination.mkdir(parents=True,exist_ok=True)
    parents = json.loads((source/'PARENTS.json').read_text())['rows']
    patches = json.loads((source/'PATCHES.json').read_text())['rows']
    labels = [('lq','LQ · V3预处理输入'),('vosr_raw','VOSR＋S2.2 GT · 对齐前'),('original_raw','原始GT · 未增粗 · 对齐前'),
              ('dilate2_raw','原始GT＋2px · 对齐前'),('background_mask','统计背景掩膜 · 白色为有效区域'),
              ('vosr_aligned','VOSR GT · 实际对比度对齐后'),('original_aligned','原始GT · 对比度对齐后'),('dilate2_aligned','原始GT＋2px · 实际对比度对齐后')]
    patch_models = [dict(key=k,label=v,note='训练目标数据，无模型推理；对齐图为float目标四舍五入至8bit供显示') for k,v in labels]
    notes = '这是用于审计GT监督差异的共同训练样本，不是Mojito039。全部19张、145个patch均保留，不按差异大小筛选。\n每块真实裁取256×256；背景mask对应审计统计的同一支持区域。\n对比度对齐后的图为实际float目标量化到8bit供看图，定量指标基于float计算。原始GT未增粗对齐是诊断反事实，未用于这两组已完成训练。\n整图版本仅展示对齐前源文件；实际亮度匹配是在每个patch内独立执行。'
    patch_manifest = dict(title='GT监督核查 · 145个实际训练patch',subtitle='19张共同训练图 · 8版本 · 同步缩放',notes=notes,
                           models=patch_models,rows=patches,defaults=','.join(k for k,v in labels))
    page = render_html(patch_manifest)
    assert page.count('for(let i=0;i<4;i++)')==1
    page = page.replace('for(let i=0;i<4;i++)','for(let i=0;i<8;i++)').replace('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;',
                'grid-template-columns:1fr 1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    (destination/'patches.html').write_text(page,encoding='utf-8')
    parent_models = [dict(key=k,label=v,note='对比度对齐前，保持源文件像素；每patch位置见总览') for k,v in labels[:4]]
    parent_manifest = dict(title='GT监督核查 · 19张共同训练图',subtitle='整图四版本 · 对齐前',notes=notes,
                           models=parent_models,rows=parents,defaults='lq,vosr_raw,original_raw,dilate2_raw')
    (destination/'parents.html').write_text(render_html(parent_manifest),encoding='utf-8')
    text = '<!doctype html><meta charset="utf-8"><title>GT监督核查：19张图与145个patch</title><style>body{font:17px/1.7 system-ui;background:#eef2f6;color:#18314b;margin:26px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{background:white;padding:18px;border-radius:10px}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:5px 10px;border:1px solid #ccd}</style><h1>GT监督核查：19张图与145个patch</h1><p>全部样本保留，不只挑差异大的。黄框为审计使用的256×256 patch。</p><p><a href="parents.html">整图四版本对比</a> · <a href="patches.html">全部145个patch八版本对比</a> · <a href="../grid039_diagnostic_20261104/index.html#training-target-evidence">返回诊断报告</a></p><p>候选来自共同训练集，不是Mojito039。对齐后的原始GT未增粗版属于诊断反事实；两组实际GT分别为VOSR对齐版和增粗GT对齐版。</p><div class="grid">'
    (destination/'thumbnails').mkdir(exist_ok=True)
    for row in parents:
        shutil.copy2(source/row['thumbnail'],destination/row['thumbnail'])
        first = row['patches'][0]
        text += '<article><h2>'+row['id']+' · '+html.escape(row['name'])+'</h2><a href="parents.html#'+row['id']+'"><img src="'+row['thumbnail']+'"></a><p><a href="parents.html#'+row['id']+'">查看整图四版本</a> · <a href="patches.html#'+first+'">查看本图patch</a></p><p>'
        for key in row['patches']:
            text += '<a href="patches.html#'+key+'">'+key+'</a> '
        text += '</p></article>'
    text += '</div><h2>全部patch指标（对齐后GT−LQ，灰阶单位）</h2><p>每行链接打开该patch；亮度和色彩均为统计背景内绝对差。色彩差不是ΔE。</p><table><tr><th>编号／来源</th><th>坐标</th><th>VOSR亮度差</th><th>增粗GT亮度差</th><th>VOSR色彩差</th><th>增粗GT色彩差</th></tr>'
    for row in patches:
        record = row['metrics']
        a,b = record['variants']['vosr']['aligned'],record['variants']['dilate2']['aligned']
        text += '<tr><td><a href="patches.html#'+row['id']+'">'+row['id']+'</a> '+html.escape(record['dataset']+'/'+record['sample_id'])+'</td><td>'+str(record['xy'])+'</td>'+''.join(f'<td>{v:.2f}</td>' for v in [a['mae_y_dn'],b['mae_y_dn'],a['mae_chroma_dn'],b['mae_chroma_dn']])+'</tr>'
    (destination/'index.html').write_text(text+'</table>',encoding='utf-8')
    save(destination/'patch_manifest.json',patch_manifest)
    save(destination/'parent_manifest.json',parent_manifest)
    for filename in ['AUDIT.json','COMPLETE.json']:
        shutil.copy2(source/filename,destination/filename)
    print(json.dumps(dict(page=str(destination/'index.html'),parents=19,patches=145,variants=8)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['extract','publish'])
    args = parser.parse_args()
    extract() if args.mode=='extract' else publish()
