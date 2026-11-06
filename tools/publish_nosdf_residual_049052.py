"""Local fixed-scale residual viewer for frozen no-SDF checkpoint probes."""
from pathlib import Path
import json
import re
import argparse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/nosdf_latest_residual_049052_20261028'
OLD=ROOT/'evaluation/v3pre_residual_frequency_049052_20261028'


def colorize(value):
    """@luna-comment Input:signed Y residual in8bit codes. Output:RGB diagnostic image. Key Steps:fixed zero-neutral interpolation. Key Params:display clip±32 only. State/Dependencies:numpy, no independent normalization."""
    import numpy as np
    t=np.clip(value/32,-1,1)[...,None]
    neutral=np.array([245,245,245],np.float32)
    red=np.array([190,20,20],np.float32);blue=np.array([20,70,210],np.float32)
    return np.rint(neutral+np.maximum(t,0)*(red-neutral)+np.maximum(-t,0)*(blue-neutral)).clip(0,255).astype(np.uint8)


def main():
    """@luna-comment Input:owned output name,24 completed records and prior ROI evidence. Output:labeled viewer/metrics with optional previous25K choices. Key Steps:fixed-scale raw/low/high rendering, shared source UI, all image sizes checked. Key Params:4 original ROIs,sigma1.4 padded filtering. State/Dependencies:owned outputs, read-only previous report."""
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name',default=OUT.name)
    args=parser.parse_args()
    assert Path(args.output_name).name==args.output_name
    OUT=ROOT/'evaluation'/args.output_name
    import cv2
    import numpy as np
    result=json.loads((OUT/'RESULT.json').read_text())
    assert result['complete'] and len(result['records'])==24
    old_metrics=json.loads((OLD/'RESIDUAL_FREQUENCY.json').read_text())['records']
    previous=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    labels={'full':'Full','text15':'Text x1.5','adapt_l1':'Adaptive-L1 cap4'}
    models=[dict(key='lq',label='V3 preprocessed input',note='Exact reused finetune LQ; no new preprocessing')]
    bindings=result['bindings']+[dict(key='sdf250_'+a,arm=a,iteration=250000,kind='sdf250') for a in labels]
    rows=[];metrics=[]
    for b in bindings:
        label=f'{labels[b["arm"]]} | '+('no SDF' if b['kind']=='nosdf' else 'SDF')+f' | {b["iteration"]//1000}K'
        for suffix,text in [('_output','RGB output'),('_residual','Raw Y residual'),('_low','Residual lowpass'),('_high','Residual highpass')]:
            models.append(dict(key=b['key']+suffix,label=label+' | '+text,note='Native256/224/crop8/smooth16; residual displays fixed +/-32 Y codes; low/high sigma1.4'))
    for original in previous['rows']:
        identity=original['id'];images={}
        for b in bindings:
            key=b['key']
            array=OLD/f'{identity}_{b["arm"]}_raw_roi.npz' if b['kind']=='sdf250' else OUT/f'{identity}_{key}_raw_roi.npz'
            data=np.load(array)
            low=cv2.GaussianBlur(data['rp'],(0,0),1.4)[32:-32,32:-32]
            assert data['r'].shape==(350,1200) and np.isfinite(data['r']).all()
            for suffix,value in [('_residual',data['r']),('_low',low),('_high',data['r']-low)]:
                name=f'{identity}_{key}{suffix}.png'
                cv2.imwrite(str(OUT/name),cv2.cvtColor(colorize(value),cv2.COLOR_RGB2BGR));images[key+suffix]=name
            images[key+'_output']=f'../{OLD.name}/{identity}_{b["arm"]}_output.png' if b['kind']=='sdf250' else f'{identity}_{key}_output.png'
            record=next(r for r in old_metrics if r['id']==identity and r['arm']==b['arm']) if b['kind']=='sdf250' else next(r for r in result['records'] if r['id']==identity and r['key']==key)
            x=data['x'];r=data['r'];ink=cv2.dilate(x,np.ones((15,15),np.uint8))-x>45
            core=cv2.erode(ink.astype(np.uint8),np.ones((3,3),np.uint8))>0
            ring=(cv2.dilate(ink.astype(np.uint8),np.ones((7,7),np.uint8))>0)&~ink
            metrics.append(dict(id=identity,key=key,iteration=b['iteration'],edge_gain=1+record['local']['gradient_residual_projection'],core_residual=float(r[core].mean()),ring_residual=float(r[ring].mean()),residual_rms=float(np.sqrt(np.mean(r*r)))))
        row=dict(id=identity,name=original['name'],width=1200,height=350,lq=f'../{OLD.name}/{identity}_input.png',images=images)
        for path in [row['lq'],*images.values()]:
            image=cv2.imread(str(OUT/path));assert image is not None and image.shape[:2]==(350,1200),path
        rows.append(row)
    iteration_label='/'.join(str(n//1000)+'K' for n in sorted({b['iteration'] for b in result['bindings']}))
    prior=ROOT/'evaluation/nosdf_latest_residual_049052_20261028'
    if OUT!=prior and (prior/'manifest.json').exists():
        earlier=json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
        metrics.extend({**m,'key':'prior25_'+m['key']} for m in json.loads((prior/'METRICS.json').read_text()) if m['key'].startswith('nosdf_'))
        for model in earlier['models']:
            if not model['key'].startswith('nosdf_'):continue
            key='prior25_'+model['key']
            models.append({**model,'key':key})
            for row in rows:
                oldrow=next(r for r in earlier['rows'] if r['id']==row['id'])
                row['images'][key]=f'../{prior.name}/'+oldrow['images'][model['key']]
    note=f'Frozen latest saved checkpoints: no-SDF arms {iteration_label}. Includes original SDF same-iteration references, SDF250K context and prior no-SDF25K when available. Red=increased Y; blue=decreased Y; fixed display range +/-32 codes. Native raw residual before RGB clamp. Four selected samples only; different runs and training stages are not strict causal ablations.'
    manifest=dict(title='Mojito049-052: no-SDF '+iteration_label+' residuals',subtitle='Frozen '+iteration_label+' | original native grid | fixed signed scale',models=models,rows=rows,defaults='lq,nosdf_full_residual,nosdf_text15_residual,nosdf_adapt_l1_residual',notes=note)
    for row in rows:
        for path in [row['lq'],*row['images'].values()]:
            image=cv2.imread(str(OUT/path));assert image is not None and image.shape[:2]==(350,1200),path
    page=(OLD/'viewer.html').read_text(encoding='utf-8')
    page=re.sub(r'<script>const DATA=.*?;</script>',lambda m:'<script>const DATA='+json.dumps(manifest,ensure_ascii=False)+';</script>',page,count=1,flags=re.S)
    page=re.sub(r'<h1>.*?</h1>','<h1>'+manifest['title']+'</h1>',page,count=1)
    page=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+note+'</div>',page,count=1,flags=re.S)
    (OUT/'viewer.html').write_text(page,encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2))
    table='<table><tr><th>Image</th><th>Model</th><th>Input-edge response</th><th>Core residual</th><th>Exterior residual</th></tr>'
    for r in metrics:
        table+=f'<tr><td>{r["id"]}</td><td>{r["key"]} {r["iteration"]//1000}K</td><td>{r["edge_gain"]:.4f}</td><td>{r["core_residual"]:+.3f}</td><td>{r["ring_residual"]:+.3f}</td></tr>'
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>No-SDF residual probe</title><style>body{font:16px/1.6 system-ui;margin:30px auto;max-width:1200px}td,th{padding:6px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Latest no-SDF residual probe</h1><p>'+note+'</p><p><a href="viewer.html">Open synchronized residual viewer</a></p>'+table+'</table><p>Input-edge response is projected output/input gradient gain on the fixed original edge-center support. These descriptive metrics are not subjective quality scores. All image assets have been decoded and verified at1200x350.</p>',encoding='utf-8')
    (OUT/'ASSET_CHECKS.json').write_text(json.dumps(dict(passed=True,rows=4,choices=len(models),image_checks=4*len(models),raw_branch_max_error=max(r['branch_error'] for r in result['records'])),indent=2))
    print(f'PASS:24 new captures;4 rows;{len(models)} choices')


if __name__=='__main__':main()
