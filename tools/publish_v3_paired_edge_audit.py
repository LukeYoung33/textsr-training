"""Publish paired-data diagnostics, separating contrast, geometry and target differences."""
from pathlib import Path
import json
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'evaluation/v3_paired_edge_audit_20261028'
TEMPLATE=ROOT/'evaluation/v3pre_residual_frequency_049052_20261028/viewer.html'


def median(values):
    """@luna-comment Input:possibly empty numeric observations. Output:median orNone. Key Steps:materialize/filter. Key Params:descriptive only. State/Dependencies:stdlib."""
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def main():
    """@luna-comment Input:completed132-record audit and original PNGs. Output:viewer,contrast-normalized diagnostics and group report. Key Steps:validate assets,derive robust contrast spans,separate aligned norm/direction,render neutral evidence. Key Params:30edge-pixel minimum,p99-p1 span. State/Dependencies:no training/source data changes."""
    import cv2
    import numpy as np
    from tools.audit_v3_paired_edges import luma
    data=json.loads((OUT/'RESULT.json').read_text());assert data['complete'] and len(data['records'])==132 and len(data['parents'])==64
    scales=json.loads((OUT/'TARGET_SCALE_AUDIT.json').read_text())
    models=[dict(key=k,label=label,note=note) for k,label,note in [
        ('lq','LQ | V3 input','Exact source-view input and baseline shard bytes'),
        ('gt','GT | V3 target','Teacher target after independent V3; not a clean captured reference'),
        ('raw_lq','LQ | before V3','Original preprocessor input; unchanged native pixels'),
        ('raw_gt','Teacher GT | before V3','Raw pseudo-target before current V3 preprocessing'),
        ('original_gt','Original GT lineage | if available','Finetune anchors have no original-GT binding: this choice falls back to raw teacher GT'),
        ('aligned_gt','GT Y | diagnostic translation','Grayscale GT with bounded local shift; diagnostic only, not a corrected dataset'),
        ('model','Full no-SDF75K | output','Actual baseline-patch inference; anchors use accepted full-frame native merge'),
        ('target_residual','GT - LQ | after V3','Red positive,blue negative; fixed +/-32 Y codes'),
        ('raw_target_residual','GT - LQ | before V3','Same fixed +/-32 code display range'),
        ('model_residual','Model - LQ | diagnostic','Signed Y after RGB quantization, not the separately captured raw residual branch')]]
    rows=[];normalized=[]
    records=sorted(data['records'],key=lambda r:(not r['key'].startswith('mojito_'),r['key']))
    for r in records:
        h,w=r['shape_hw'];images={}
        for m in models:
            name=r['key']+'_'+m['key']+'.png';image=cv2.imread(str(OUT/name),cv2.IMREAD_UNCHANGED)
            assert image is not None and image.shape[:2]==(h,w),name
            images[m['key']]=name
        identifier=r['key'].replace('mojito_','')
        xy=r['metadata'].get('xy',r['metadata'].get('display_roi'))
        rows.append(dict(id=identifier,name=r['dataset']+' | '+r['sample_id']+' | '+str(xy),width=w,height=h,lq=images['lq'],images={k:v for k,v in images.items() if k!='lq'}))
        n=dict(key=r['key'],dataset=r['dataset'],pairs={})
        for pair,xkey,ykey,alignkey in [('raw_pair','raw_lq','raw_gt','raw_alignment'),('v3_pair','lq','gt','alignment')]:
            x=luma(cv2.imread(str(OUT/images[xkey])));y=luma(cv2.imread(str(OUT/images[ykey])))
            x=x[16:-16,16:-16];y=y[16:-16,16:-16]
            sx=float(np.percentile(x,99)-np.percentile(x,1));sy=float(np.percentile(y,99)-np.percentile(y,1))
            ratio=sy/max(sx,1e-12)
            item=dict(reference_span_p99_p1=sx,target_span_p99_p1=sy,contrast_ratio=ratio,norm_ratio=r[pair]['gradient_norm_ratio']/max(ratio,1e-12),gain=r[pair]['gradient_gain']/max(ratio,1e-12),edge_pixels=r[pair]['edge_pixels'])
            alignment=r[alignkey]
            if alignment['available']:
                item['aligned_norm_ratio']=alignment['metrics']['gradient_norm_ratio']/max(ratio,1e-12)
                item['aligned_gain']=alignment['metrics']['gradient_gain']/max(ratio,1e-12)
            n['pairs'][pair]=item
        normalized.append(n)
    groups=[]
    for group in ('HXA','HXB','lenovo0926','lenovo1011','mojito_finetune_anchor'):
        rs=[r for r in records if r['dataset']==group];eligible=[r for r in rs if r['v3_pair']['edge_pixels']>=30]
        nr=[r for r in normalized if r['dataset']==group and r['pairs']['v3_pair']['edge_pixels']>=30]
        groups.append(dict(dataset=group,patches=len(rs),edge_eligible=len(eligible),raw_target_gain=median(r['raw_pair']['gradient_gain'] for r in eligible),v3_target_gain=median(r['v3_pair']['gradient_gain'] for r in eligible),v3_target_norm=median(r['v3_pair']['gradient_norm_ratio'] for r in eligible),v3_target_cosine=median(r['v3_pair']['gradient_cosine'] for r in eligible),aligned_v3_gain=median(r['alignment']['metrics']['gradient_gain'] for r in eligible if r['alignment']['available']),contrast_normalized_norm=median(r['pairs']['v3_pair']['norm_ratio'] for r in nr),aligned_contrast_normalized_norm=median(r['pairs']['v3_pair'].get('aligned_norm_ratio') for r in nr),raw_contrast_normalized_norm=median(r['pairs']['raw_pair']['norm_ratio'] for r in nr),target_softening_sign_count=sum(r['v3_pair']['core']['mean'] is not None and r['v3_pair']['ring']['mean'] is not None and r['v3_pair']['core']['mean']>0 and r['v3_pair']['ring']['mean']<0 for r in rs),residual_highpass_cosine=median(r['target_model_highpass_residual_cosine'] for r in rs)))
    note='Data audit:64 seeded stratified pretrain parents,128 baseline text patches and4 Mojito finetune anchors. All sampled bank LQ/GT bytes exactly match V3 source crops; source hashes verified. Mojito anchors are not current pretraining examples. GT is a teacher pseudo-target; original-GT lineage is shown only where bound. Translation/contrast normalization are diagnostic, not dataset corrections. All signed displays use +/-32 Y codes. Stopped runs remain stopped.'
    manifest=dict(title='GT / LQ data-side edge audit',subtitle='Raw pairs | V3 pairs | GT targets | no-SDF75K',models=models,rows=rows,defaults='lq,gt,target_residual,model',notes=note)
    page=TEMPLATE.read_text(encoding='utf-8')
    page=re.sub(r'<script>const DATA=.*?;</script>',lambda m:'<script>const DATA='+json.dumps(manifest,ensure_ascii=False)+';</script>',page,count=1,flags=re.S)
    page=re.sub(r'<h1>.*?</h1>','<h1>'+manifest['title']+'</h1>',page,count=1)
    page=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+note+'</div>',page,count=1,flags=re.S)
    (OUT/'viewer.html').write_text(page,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    analysis=dict(groups=groups,contrast_normalization=normalized,span_definition='p99 minus p1 on fixed16px interior; contrast adjustment is a diagnostic approximation, not an MTF or intrinsic sharpness estimate',geometry_definition='bounded gradient-cosine translation; refined shift is not ground-truth registration; cubic interpolation affects gradients',sample_scope='64 parent-balanced draws,128 correlated patches,4 separately labeled finetune anchors; no population inference')
    (OUT/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2))
    body='''<!doctype html><meta charset="utf-8"><title>Paired-data edge audit</title><style>body{font:16px/1.65 system-ui;max-width:1280px;margin:30px auto;padding:0 20px;color:#18283a}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:7px;border-bottom:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}section{background:#f4f7fa;padding:20px;margin:20px 0}a{color:#006da3}</style><h1>Paired-data edge audit</h1>'''
    body+='<p>'+note+'</p><p><a href="viewer.html#049">Open synchronized data viewer at049</a> · <a href="ANALYSIS.json">Full diagnostics</a> · <a href="RESULT.json">Source/hash/probe receipt</a> · <a href="TARGET_SCALE_AUDIT.json">Full target-scale audit</a></p>'
    body+='<section><h2>Upstream target scale:1221 of4873 training parents use roughly2x cubic enlargement</h2><p>This occurs in target refresh before V3 preprocessing:the archived teacher output has approximately half the native LQ width/height and is resized to match LQ. For all16 sampled HXB parents, replaying INTER_CUBIC reproduces the pre-V3 training target exactly (maximum pixel error0). This is not a claim that all4873 targets are bad; it identifies an inconsistent source-resolution binding that deserves review.</p><table><tr><th>Training dataset</th><th>Original-size teacher targets</th><th>Cubic-enlarged teacher targets</th></tr>'
    for group in ('HXA','HXB','lenovo0926','lenovo1011'):
        identity=sum(s['parents'] for s in scales['summary']['train'] if s['dataset']==group and s['interpolation']=='identity')
        resized=sum(s['parents'] for s in scales['summary']['train'] if s['dataset']==group and s['interpolation']=='INTER_CUBIC')
        body+=f'<tr><td>{group}</td><td>{identity}</td><td>{resized}</td></tr>'
    body+='</table><p>Source:tools/manage_vosr_2u_block256.py::source_parent, target-only cv2.resize(...,INTER_CUBIC). Full validation583 intentionally retains older GT_VOSR and is separately labeled in its existing manifests. Counts here refer to parents, not weighted patch sampling probabilities. Cubic enlargement provides interpolation, not native captured high-frequency detail.</p></section>'
    body+='<section><h2>Directly verified</h2><p>All three no-SDF runs and their controller were stopped by user request; saved25K/50K/75K artifacts are retained. No dataset rewriting or replacement training was started.</p><p>128 selected baseline LQ/GT patches match the declared preprocessed source crops byte-for-byte. Original and preprocessed source hashes and sampled bank receipts were verified. No sampled baseline crop/channel swap was detected.</p><p>96 of128 selected pretraining patches have a positive inner-core and negative exterior-ring target residual even with no model or SDF involved. Raw teacher targets already have lower projected input-edge responses in group medians; current V3 does not introduce the entire discrepancy.</p><p>049 shows character/stroke-shape disagreement in the raw teacher GT that persists after V3 and bounded translation. This is a separate finetune pseudo-target issue; it cannot by itself explain the pretrain75K model. The viewer exposes the actual pixels for review.</p></section>'
    body+='<section><h2>Pretraining targets: edge magnitude and direction separated</h2><p>Medians on patches with at least30 input edge pixels. Projection gain combines norm and direction; low projection alone is not proof of blur. Norms are measured on fixed input-edge positions, not all target edges.</p><table><tr><th>Dataset</th><th>Eligible / sampled</th><th>Raw target gain</th><th>V3 target gain</th><th>V3 gradient norm ratio</th><th>V3 direction cosine</th><th>Shift-probed gain</th><th>Softening signs</th></tr>'
    for g in groups:
        body+=f'<tr><td>{g["dataset"]}</td><td>{g["edge_eligible"]}/{g["patches"]}</td><td>{g["raw_target_gain"]:.3f}</td><td>{g["v3_target_gain"]:.3f}</td><td>{g["v3_target_norm"]:.3f}</td><td>{g["v3_target_cosine"]:.3f}</td><td>{g["aligned_v3_gain"]:.3f}</td><td>{g["target_softening_sign_count"]}/{g["patches"]}</td></tr>'
    body+='</table></section><section><h2>Contrast control</h2><p>Divide gradient norm ratios by each image pair’s robust p99−p1 luma-span ratio. This reduces a global contrast explanation but does not isolate an optical edge-spread function. The aligned variant uses the original robust span; shift interpolation itself can change gradient magnitude.</p><table><tr><th>Dataset</th><th>Raw contrast-adjusted norm</th><th>V3 contrast-adjusted norm</th><th>Shift + contrast-adjusted norm</th></tr>'
    for g in groups:body+=f'<tr><td>{g["dataset"]}</td><td>{g["raw_contrast_normalized_norm"]:.3f}</td><td>{g["contrast_normalized_norm"]:.3f}</td><td>{g["aligned_contrast_normalized_norm"]:.3f}</td></tr>'
    body+='</table></section><section><h2>Interpretation and limits</h2><p>The measured target data already ask for edge attenuation or spatial/style changes on many samples. A pixel-reconstruction model can learn those changes and average inconsistent edge targets, even without SDF. This is a supported data-side mechanism, not proof of a unique historical cause or a judgment of all teacher outputs.</p><p>Small bounded translations improve some patches but do not eliminate group-level target/input differences. Character-shape changes, local distortions, teacher processing and contrast remain distinct from a single global translation. Raw-versus-V3 checks indicate that replacing or fixing only V3 is not justified by these data.</p><p>Further action should first review teacher pseudo-targets and their generation lineage; no GT regeneration, rejection rule, geometry correction or retraining has been applied. Subjective image acceptance remains with the user.</p></section>'
    (OUT/'index.html').write_text(body,encoding='utf-8')
    (OUT/'ASSET_CHECKS.json').write_text(json.dumps(dict(passed=True,rows=len(rows),choices=len(models),decoded_images=len(rows)*len(models)),indent=2))
    print(json.dumps(dict(groups=groups,rows=len(rows),choices=len(models)),indent=2))


if __name__=='__main__':main()
