"""Portable old/new document-pipeline comparisons, intermediate viewers and validation."""
import base64
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_offline_camera_matrix import ARMS,sha,save_json,verify_frozen
from tools.evaluate_doc_warp_matrix import BASE,OUT,INVENTORY,document_record,source_path
from tools.doc_warp_composition import inverse_residual

COMPARISON_CONFIG=OUT/'comparison_config.json'
COMPARE=Path(json.loads(COMPARISON_CONFIG.read_text())['previous_root']) if COMPARISON_CONFIG.exists() else BASE
NAMES=dict(zip(ARMS,('V5 real','V3 real','V5 synthetic','V3 synthetic')))


@lru_cache(maxsize=64)
def embedded(path):
    """@luna-comment
    Input: existing task/baseline image path. Output: offline JPEG data URI.
    Key Steps: decode, proportional area resize, JPEG85/base64 encode with bounded cache.
    Key Params:700px longest edge; full native files remain linked separately.
    State/Dependencies: read-only image assets and per-worker64-entry string cache.
    """
    image=cv2.imread(str(path))
    if image is None:
        raise ValueError('Missing gallery image: '+str(path))
    factor=min(1.,700/max(image.shape[:2]))
    image=cv2.resize(image,(round(image.shape[1]*factor),round(image.shape[0]*factor)),interpolation=cv2.INTER_AREA)
    okay,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not okay:
        raise IOError('Gallery image encoding failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode('ascii')


def figure(label,path,link):
    """@luna-comment
    Input: display label/image path/native link. Output: escaped embedded figure.
    Key Steps: create safe caption and image/native anchor. Key Params: no asset fetch.
    State/Dependencies: embedded preview helper and immutable native image.
    """
    return f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{html.escape(link,quote=True)}" target="_blank"><img src="{embedded(str(path))}"></a></figure>'


def shell(title,body):
    """@luna-comment
    Input: escaped/constructed page body and title. Output: standalone responsive HTML.
    Key Steps: wrap shared styles and metadata. Key Params: paired comparison columns.
    State/Dependencies: inline styles only; no external resources.
    """
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{font:15px/1.6 Segoe UI,sans-serif;color:#183247;background:#f1f5f9;max-width:1500px;margin:24px auto;padding:0 20px}h1{font-size:29px;line-height:1.3}h2{font-size:21px}section{background:white;border-radius:12px;padding:20px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.stages{grid-template-columns:repeat(3,minmax(0,1fr))}figure{margin:0;min-width:0}figcaption{font-weight:600;margin-bottom:8px}img{width:100%;height:340px;object-fit:contain;background:#e7edf2}a{color:#1a607b;text-decoration:none}nav{display:flex;align-items:center;gap:16px;flex-wrap:wrap}nav a,select{padding:9px;border:1px solid #c2d2df;border-radius:7px;background:white;color:#23556c}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.6 Consolas,monospace;background:#edf3f7;padding:15px}.note{padding:14px;background:#e7f2ef;border-radius:8px}.muted{color:#6d8091;font-size:13px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e0e7ed;padding:9px;text-align:left}@media(max-width:800px){.grid,.stages{grid-template-columns:1fr}body{padding:0 10px}img{height:300px}}</style>'+body+'</html>'


def document_preview(row,record):
    """@luna-comment
    Input: original row and unchanged released quad predictions.
    Output: display-resolution overlay with visible three-pixel document outlines.
    Key Steps: reuse original preview, map decoded source coordinates into display
    coordinates, redraw identical quads without altering native detector artifacts.
    Key Params: preview-only line thickness3; geometry/model unchanged.
    State/Dependencies: baseline preview, decoded source dimensions and own JPEG output.
    """
    image=cv2.imread(str(BASE/'previews'/(row['id']+'_source.jpg')))
    if image is None:
        raise ValueError('Missing original display preview')
    scale=np.float64([image.shape[1]/record['image_width'],image.shape[0]/record['image_height']])
    for item in record['instances']:
        color=(40,200,40) if item['geometry_type']=='boundary_quad' else (255,210,0)
        points=np.rint(np.asarray(item['quad'])*scale).astype(np.int32)
        cv2.polylines(image,[points],True,color,3,cv2.LINE_AA)
        position=tuple(np.clip(points[0]+[0,17],[0,15],[image.shape[1]-50,image.shape[0]-5]))
        cv2.putText(image,str(item['instance_id'])+f" {item['confidence']:.2f}",position,cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    path=OUT/'previews'/(row['id']+'_documents.jpg')
    path.parent.mkdir(exist_ok=True)
    if not cv2.imwrite(str(path),image):
        raise IOError('Document display preview write failed')
    return path


def intermediate_page(row,item,record_directory):
    """@luna-comment
    Input: original row/released document instance. Output: all-arm intermediate viewer.
    Key Steps: embed shared warp/support/OCR/pre, arm-select model/post/unwarp/alpha;
    display exact source-to-warp/inverse matrices and diagnostic geometry.
    Key Params: native file links retained; alpha preview is diagnostic8bit.
    State/Dependencies: completed stage receipts and read-only frozen/baseline outputs.
    """
    identifier=f"{item['instance_id']:03d}"
    ocr=json.loads((OUT/'ocr'/row['id']/(identifier+'.json')).read_text())
    prep=json.loads((OUT/'prepared'/row['id']/(identifier+'.receipt.json')).read_text())
    shared=[('Original document warp',record_directory/item['image_file'],f"../documents/{record_directory.name}/{item['image_file']}"),
            ('Observed source support',OUT/ocr['valid_file'],'../'+ocr['valid_file']),
            ('Fresh warped text detection',OUT/ocr['overlay'],'../'+ocr['overlay']),
            ('Preprocessed warp · gamma1.8',OUT/prep['image'],'../'+prep['image'])]
    body='<h1>Document intermediate stages</h1><p><a href="../pages/'+row['id']+'.html">← Full-frame comparison</a></p><p>'+html.escape(row['id']+' · '+row['name'])+f" · Instance {identifier} · {html.escape(item['geometry_type'])}</p>"
    body+='<section><div class="grid stages">'+''.join(figure(label,path,link) for label,path,link in shared)+'</div></section>'
    arm_data={}
    for arm in ARMS:
        stage=json.loads((OUT/arm/'instances'/row['id']/(identifier+'.json')).read_text())
        arm_data[arm]={key:dict(image=embedded(str(OUT/path)),link='../'+path) for key,path in stage['paths'].items()}
    labels={'model':'Native text model output','post':'Postprocessed warp · gamma1.2','unwarp':'Unwarp composite after this instance','alpha':'Inverse mapped contribution alpha'}
    options=''.join(f'<option value="{arm}">{NAMES[arm]}</option>' for arm in ARMS)
    body+='<section><h2>Model-specific stages</h2><select id="stageArm" onchange="showStageArm(this.value)">'+options+'</select><div class="grid">'
    for key,label in labels.items():
        value=arm_data[ARMS[0]][key]
        body+=f'<figure><figcaption>{label}</figcaption><a id="link-{key}" href="{value["link"]}" target="_blank"><img id="img-{key}" src="{value["image"]}"></a></figure>'
    body+='</div></section>'
    diagnostic=dict(source_sha256=row['sha256'],instance_id=item['instance_id'],confidence=item['confidence'],quad=item['quad'],
                    geometry_type=item['geometry_type'],warp=item['warp'],inverse=ocr['inverse'],roundtrip_error=ocr['roundtrip_error'],
                    warped_text_boxes=len(ocr['boxes']),unknown_padding_contributes=False,overlap_policy='descending confidence / stable first ownership')
    body+='<details><summary>Coordinates, transforms and diagnostics</summary><pre>'+html.escape(json.dumps(diagnostic,indent=2))+'</pre></details>'
    payload=json.dumps(arm_data,separators=(',',':')).replace('<','\\u003c')
    body+='<script>const stageData='+payload+';\n// @luna-comment Input: selected arm. Output: four stage previews/native links. Key Steps: set embedded src/link. Key Params: fixed arm keys. State/Dependencies: embedded stageData and DOM.\nfunction showStageArm(arm){for(const [key,value] of Object.entries(stageData[arm])){document.getElementById("img-"+key).src=value.image;document.getElementById("link-"+key).href=value.link;}}</script>'
    path=OUT/'stages'/(row['id']+'_'+identifier+'.html')
    path.write_text(shell('Document stages '+row['id']+'/'+identifier,body),encoding='utf-8')
    return '../stages/'+path.name


def render_frame(job):
    """@luna-comment
    Input: one original row plus full navigation inventory. Output: paired old/new page.
    Key Steps: bind source/checkpoint identities; embed original/detection and four
    baseline/corrected pairs; link all per-instance intermediate pages.
    Key Params: no-document/rejection/downscale states shown explicitly.
    State/Dependencies: independent per-frame writes; CPU worker image cache;
    optional output-local comparison_config.json selects an immutable previous run.
    """
    row,rows=job
    cv2.setNumThreads(1)
    record,directory=document_record(row)
    text_boxes=sum(len(json.loads((OUT/'ocr'/row['id']/(f"{item['instance_id']:03d}"+'.json')).read_text())['boxes']) for item in record['instances'])
    index=next(i for i,candidate in enumerate(rows) if candidate['id']==row['id'])
    previous=rows[max(0,index-1)]['id']+'.html';following=rows[min(len(rows)-1,index+1)]['id']+'.html'
    options=''.join(f'<option value="{candidate["id"]}.html"'+(' selected' if candidate['id']==row['id'] else '')+'>'+html.escape(candidate['id']+' · '+candidate['name'])+'</option>' for candidate in rows)
    body='<h1>Document warp / unwarp · Four-model comparison</h1><p>Same280 originals and same final250K checkpoints. Fresh warped OCR; preprocessing gamma1.8; native96×480 restoration; postprocessing gamma1.2; inverse residual pasteback.</p>'
    body+=f'<nav><a href="{previous}">← Previous</a><select onchange="location.href=this.value">{options}</select><a href="{following}">Next →</a><span>{index+1} / {len(rows)}</span><a href="../SUMMARY.html">Run summary</a></nav>'
    body+='<h2>'+html.escape(row['id']+' · '+row['name'])+'</h2>'
    body+=f'<p class="note">Documents: {len(record["instances"])} · Boundary fits: {record["boundary_quad_count"]} · Approximate quads: {record["approximate_quad_count"]} · Rejected: {len(record["rejected"])} · Warped text boxes: {text_boxes}</p>'
    if not record['instances']:
        body+='<p class="note">No valid document: all new-pipeline outputs preserve the original unchanged. This is not a successful rectification.</p>'
    elif text_boxes==0:
        body+='<p class="note">Valid document, but no text detected after warp: no text enhancement is applied and the original frame remains unchanged.</p>'
    original=figure('Original',BASE/'previews'/(row['id']+'_source.jpg'),'../../camera_final_20261021/source/'+source_path(row).name)
    overlay=figure('Document detection · green boundary / cyan approximate',document_preview(row,record),'../documents/'+directory.name+'/overlay.png')
    body+='<section><div class="grid">'+original+overlay+'</div></section>'
    for arm in ARMS:
        old=json.loads((COMPARE/arm/'receipts'/(row['id']+'.json')).read_text())
        new=json.loads((OUT/arm/'receipts'/(row['id']+'.json')).read_text())
        if old['source_sha256']!=row['sha256'] or new['source_sha256']!=row['sha256'] or old['checkpoint_sha256']!=new['checkpoint_sha256']:
            raise ValueError('Old/new source or checkpoint mismatch')
        body+='<section><h2>'+NAMES[arm]+'</h2><div class="grid">'
        label='Previous · no document warp' if COMPARE==BASE else 'Previous · document warp v1'
        body+=figure(label,COMPARE/arm/'previews'/(row['id']+'.jpg'),'../../'+COMPARE.name+'/'+old['final'])
        body+=figure('New · document warp → text pipeline → unwarp',OUT/arm/'previews'/(row['id']+'.jpg'),'../'+new['final'])
        body+='</div><p class="muted">'+html.escape(new['status'])+' · Same checkpoint '+new['checkpoint_sha256'][:16]+'</p></section>'
    body+='<section><h2>Preserved intermediate stages</h2>'
    for item in record['instances']:
        link=intermediate_page(row,item,directory)
        body+=f'<p><a href="{link}">Instance {item["instance_id"]:03d} · {html.escape(item["geometry_type"])} · confidence {item["confidence"]:.3f} · {item["warp"]["output_width"]}×{item["warp"]["output_height"]}</a></p>'
    body+='<p><a href="../documents/'+directory.name+'/result.json">Full document geometry and rejection record</a></p></section>'
    path=OUT/'pages'/(row['id']+'.html')
    page=shell('Warp comparison '+row['id'],body)
    path.write_text(page,encoding='utf-8')
    if index==0:
        (OUT/'index.html').write_text(page.replace('<meta charset="utf-8">','<meta charset="utf-8"><base href="pages/">',1),encoding='utf-8')
    return dict(id=row['id'],documents=len(record['instances']),boundary=record['boundary_quad_count'],approximate=record['approximate_quad_count'],
                rejected=len(record['rejected']),downscaled=sum(item['warp']['output_downscaled'] for item in record['instances']),
                padded=sum(item['warp']['padding_applied'] for item in record['instances']),text_boxes=sum(stage['text_boxes'] for stage in new['instances']))


def validate(rows):
    """@luna-comment
    Input: full original inventory and completed artifact tree. Output: acceptance proof.
    Key Steps: check release/frozen/checkpoint identities,1120 receipts, all stage links,
    all H roundtrips, representative PNG hashes and actual identity residual.
    Key Params:20 representative native results and .01px matrix bound.
    State/Dependencies: read-only outputs; writes only validation receipt at end.
    """
    frozen=verify_frozen()
    release=ROOT/'doc_release'
    assert sha(release/'best.pt')=='1921fbad80063489fed5b87428a4c28ce6132482feba3c5fd8a0d5b9caa36c77'
    assert sha(release/'doc_quad_infer.py')=='2106c82547263bb2faa85cd82865c7207063a8784235837869c4f292af9f2d87'
    count=0;roundtrip=0.;instances=0;identity_checked=False
    for arm in ARMS:
        baseline=json.loads((BASE/arm/'COMPLETE.json').read_text())
        actual=json.loads((OUT/arm/'COMPLETE.json').read_text())
        assert actual['images']==len(rows) and actual['checkpoint_sha256']==baseline['checkpoint_sha256']
        for row in rows:
            receipt=json.loads((OUT/arm/'receipts'/(row['id']+'.json')).read_text())
            assert receipt['source_sha256']==row['sha256'] and receipt['checkpoint_sha256']==baseline['checkpoint_sha256']
            assert receipt['frozen_manifest_sha256']==frozen and receipt['outside_owned_support_unchanged']
            assert receipt['source_hw']==json.loads((BASE/'prepared'/(row['id']+'.receipt.json')).read_text())['source_hw']
            assert (OUT/receipt['final']).is_file()
            for stage in receipt['instances']:
                assert stage['postprocess_gamma']==1.2 and stage['geometry']['roundtrip_error']<=.01
                roundtrip=max(roundtrip,stage['geometry']['roundtrip_error'])
                assert all((OUT/path).is_file() for path in stage['paths'].values())
                identifier=f"{stage['instance_id']:03d}"
                ocr=json.loads((OUT/'ocr'/row['id']/(identifier+'.json')).read_text())
                assert ocr['fresh_warp_detection'] and ocr['source_sha256']==row['sha256']
                if arm==ARMS[0]:
                    instances+=1
                if not identity_checked:
                    source=cv2.imread(str(source_path(row)));warp=cv2.imread(ocr['warp_file'])
                    valid=cv2.imread(str(OUT/ocr['valid_file']),cv2.IMREAD_GRAYSCALE)
                    record,_=document_record(row);item=next(item for item in record['instances'] if item['instance_id']==stage['instance_id'])
                    delta,_,_,_=inverse_residual(source,warp,warp,item['warp']['image_to_output'],item['quad'],valid,np.zeros(source.shape[:2],bool))
                    assert not np.any(delta)
                    identity_checked=True
            if row['id'] in ('001','002','089','093','280'):
                path=OUT/receipt['final'];assert sha(path)==receipt['final_sha256']
                pixels=cv2.imread(str(path));source=cv2.imread(str(source_path(row)))
                owned=cv2.imread(str(OUT/arm/'ownership'/(row['id']+'.png')),cv2.IMREAD_GRAYSCALE)>0
                assert pixels.shape==source.shape and np.array_equal(pixels[~owned],source[~owned])
                if receipt['documents']==0:
                    assert np.array_equal(pixels,source)
            assert (OUT/'pages'/(row['id']+'.html')).is_file()
            count+=1
    proof=dict(passed=True,images=len(rows),new_outputs=count,baseline_comparison_outputs=count,valid_document_instances=instances,
               max_corner_roundtrip_error=roundtrip,actual_identity_residual_checked=identity_checked,
               representative_native_pngs_checked=20,unchanged_exterior=True,exact_baseline_checkpoints=True,
               frozen_manifest_sha256=frozen,document_weights_sha256=sha(release/'best.pt'))
    save_json(OUT/'VALIDATED.json',proof)
    return proof


def main():
    """@luna-comment
    Input: completed four-arm corrected pipeline. Output: portable pages/summary/validation.
    Key Steps: parallel per-frame rendering, aggregate detector outcome statistics,
    validate exact old/new pairings and preserved stages before completion receipt.
    Key Params: six CPU gallery workers and all280 original IDs.
    State/Dependencies: output only; immutable baseline/native results unchanged.
    """
    rows=json.loads(INVENTORY.read_text())['images']
    for name in ('pages','stages'):
        (OUT/name).mkdir(exist_ok=True)
    records=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for count,record in enumerate(pool.map(render_frame,[(row,rows) for row in rows]),1):
            records.append(record)
            save_json(OUT/'gallery_progress.json',dict(done=count,expected=len(rows)))
            print('GALLERY',count,len(rows),flush=True)
    totals={key:sum(record[key] for record in records) for key in ('documents','boundary','approximate','rejected','downscaled','padded','text_boxes')}
    totals['no_document_frames']=sum(record['documents']==0 for record in records)
    totals['document_frames_without_text']=sum(record['documents']>0 and record['text_boxes']==0 for record in records)
    save_json(OUT/'DOCUMENT_STATISTICS.json',dict(totals=totals,frames=records))
    body='<h1>Document pipeline · Evaluation summary</h1><p><a href="index.html">Open paired comparisons</a></p><p>280 source photos · Four identical final250K checkpoints · 1120 new full frames paired with1120 immutable prior results.</p><pre>'+html.escape(json.dumps(totals,indent=2))+'</pre><p>Boundary/approximate status is detector output, not physical-corner ground truth. No valid document means unchanged source. Allocation limits8192 /33554432 preserve native text scale where possible; actual downscaling is recorded above. No paired GT or model-quality ranking is asserted.</p>'
    body+='<h2>Source cases</h2><table><tr><th>Image</th><th>Documents</th><th>Boundary</th><th>Approximate</th><th>Rejected</th><th>Warped text boxes</th></tr>'
    for record in records:
        body+=f'<tr><td><a href="pages/{record["id"]}.html">{record["id"]}</a></td><td>{record["documents"]}</td><td>{record["boundary"]}</td><td>{record["approximate"]}</td><td>{record["rejected"]}</td><td>{record["text_boxes"]}</td></tr>'
    (OUT/'SUMMARY.html').write_text(shell('Document pipeline summary',body+'</table>'),encoding='utf-8')
    proof=validate(rows)
    save_json(OUT/'COMPLETE.json',dict(complete=True,**proof))


if __name__=='__main__':
    main()
