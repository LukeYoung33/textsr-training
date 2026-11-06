"""Local RGB/raw-residual viewer for the four latest S2.2 Full checkpoints."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'evaluation/s22_full_four_mojito049052_20261029'
OLD=ROOT/'evaluation/v3pre_residual_frequency_049052_20261028'
PRIOR=ROOT/'evaluation/nosdf_75k_residual_049052_20261028'


def main():
    """@luna-comment Input:16 frozen capture records/local arrays and prior75K context. Output:21-choice RGB/residual viewer and descriptive summary. Key Steps:hash/shape verification,fixed±32code colors,padded sigma1.4 bands,shared unchanged viewer. Key Params:actual distinct checkpoint iterations. State/Dependencies:owned publication only;no subjective quality ranking."""
    import cv2
    import numpy as np
    from tools.publish_nosdf_residual_049052 import colorize
    result=json.loads((OUT/'RESULT.json').read_text());assert result['complete'] and len(result['records'])==16
    old_manifest=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    prior_records=json.loads((PRIOR/'RESULT.json').read_text())['records']
    labels={'paired_full':'S2.2 | Real pairs Full','gt_degrade30_full':'S2.2 | 30% GT degradation Full','identity_full':'S2.2 | GT identity Full','guard_full':'S2.2 | Trusted edge guard Full','previous75':'Previous GT | no-SDF Full'}
    bindings=result['bindings']+[dict(key='previous75',iteration=75000)]
    models=[dict(key='lq',label='V3 preprocessed input',note='Exact original V3 input;no repeat preprocessing')]
    for b in bindings:
        for suffix,caption in [('_output','RGB output'),('_residual','Raw Y residual'),('_low','Residual lowpass'),('_high','Residual highpass')]:
            models.append(dict(key=b['key']+suffix,label=labels[b['key']]+f' | {b["iteration"]//1000}K | '+caption,note='Native256/224/batch8/crop8/smooth16; raw Y before RGB clipping; fixed residual display +/-32codes; sigma1.4 low/high'))
    rows=[];metrics=[]
    for original in old_manifest['rows']:
        identity=original['id'];images={}
        for b in bindings:
            key=b['key'];prior=key=='previous75'
            array=PRIOR/f'{identity}_nosdf_full_raw_roi.npz' if prior else OUT/f'{identity}_{key}_raw_roi.npz'
            record=next(r for r in prior_records if r['id']==identity and r['key']=='nosdf_full') if prior else next(r for r in result['records'] if r['id']==identity and r['key']==key)
            assert hashlib.sha256(array.read_bytes()).hexdigest()==record['artifact_sha256']
            data=np.load(array);assert data['r'].shape==(350,1200) and np.isfinite(data['r']).all()
            low=cv2.GaussianBlur(data['rp'],(0,0),1.4)[32:-32,32:-32]
            for suffix,value in [('_residual',data['r']),('_low',low),('_high',data['r']-low)]:
                name=f'{identity}_{key}{suffix}.png'
                assert cv2.imwrite(str(OUT/name),cv2.cvtColor(colorize(value),cv2.COLOR_RGB2BGR));images[key+suffix]=name
            output=f'../{PRIOR.name}/{identity}_nosdf_full_output.png' if prior else f'{identity}_{key}_output.png'
            images[key+'_output']=output
            if not prior:assert hashlib.sha256((OUT/output).read_bytes()).hexdigest()==record['output_sha256']
            x=data['x'];r=data['r'];ink=cv2.dilate(x,np.ones((15,15),np.uint8))-x>45
            core=cv2.erode(ink.astype(np.uint8),np.ones((3,3),np.uint8))>0
            ring=(cv2.dilate(ink.astype(np.uint8),np.ones((7,7),np.uint8))>0)&~ink
            metrics.append(dict(id=identity,key=key,iteration=b['iteration'],edge_gain=1+record['local']['gradient_residual_projection'],core_residual=float(r[core].mean()),ring_residual=float(r[ring].mean()),residual_rms=float(np.sqrt(np.mean(r*r)))))
        row=dict(id=identity,name=original['name'],width=1200,height=350,lq=f'../{OLD.name}/{identity}_input.png',images=images)
        for path in [row['lq'],*images.values()]:
            image=cv2.imread(str(OUT/path));assert image is not None and image.shape[:2]==(350,1200),path
        rows.append(row)
    note='Four Full strategies only: real pairs,30% GT degradation,GT identity,trusted edge guard. 100% degradation excluded. Frozen latest saved checkpoints have different iterations;labels are exact. Same049-052 V3 inputs/original grid and fixed +/-32 residual scale. New train GT is full VOSR + S2.2; inputs receive no extra processing. Prior no-SDF75K Full is historical context,not a controlled single-variable comparison. Four-image descriptive probe;quality remains user-reviewed.'
    defaults=','.join(b['key']+'_output' for b in result['bindings'])
    manifest=dict(title='S2.2 Full: four strategies on Mojito049-052',subtitle='Latest frozen checkpoints | RGB / raw Y residual',models=models,rows=rows,defaults=defaults,notes=note)
    page=(OLD/'viewer.html').read_text(encoding='utf-8')
    page=re.sub(r'<script>const DATA=.*?;</script>',lambda m:'<script>const DATA='+json.dumps(manifest,ensure_ascii=False)+';</script>',page,count=1,flags=re.S)
    page=re.sub(r'<h1>.*?</h1>','<h1>'+manifest['title']+'</h1>',page,count=1)
    page=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+note+'</div>',page,count=1,flags=re.S)
    (OUT/'viewer.html').write_text(page,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2))
    summary=[]
    for b in bindings:
        records=[r for r in metrics if r['key']==b['key']]
        summary.append(dict(key=b['key'],label=labels[b['key']],iteration=b['iteration'],**{name:statistics.mean(r[name] for r in records) for name in ('edge_gain','core_residual','ring_residual','residual_rms')}))
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    body='<!doctype html><meta charset="utf-8"><title>S2.2 Full four-model validation</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}a{color:#006b9f}</style><h1>S2.2 Full four-model validation</h1><p>'+note+'</p><p><a href="viewer.html#049">Open synchronized RGB/residual viewer</a> · <a href="BINDINGS.json">Frozen checkpoint bindings</a></p><table><tr><th>Strategy</th><th>Checkpoint</th><th>Mean input-edge response</th><th>Core residual codes</th><th>Exterior residual codes</th></tr>'
    for row in summary:body+=f'<tr><td>{row["label"]}</td><td>{row["iteration"]//1000}K</td><td>{row["edge_gain"]:.4f}</td><td>{row["core_residual"]:+.3f}</td><td>{row["ring_residual"]:+.3f}</td></tr>'
    body+='</table><p>Response is a projection onto fixed input edge gradients. A value near1 measures preservation of that projection;it is not a subjective quality ranking. Core-positive/exterior-negative residual can include contrast,shape and alignment effects. Actual stage differences must be considered.</p>'
    (OUT/'index.html').write_text(body,encoding='utf-8')
    (OUT/'ASSET_CHECKS.json').write_text(json.dumps(dict(passed=True,captures=16,rows=4,choices=len(models),decoded_images=4*len(models),verified_raw_hashes=20,verified_new_output_hashes=16,max_branch_error=max(r['branch_error'] for r in result['records'])),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
