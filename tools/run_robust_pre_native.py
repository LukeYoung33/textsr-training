"""Accepted perspective geometry -> latest blended preprocessing -> native-frame correction."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_adaptive_pre_v21_camera_batch import sha, save_json
from tools.run_doc_pre_v21_camera import write_image, preview
from tools.adaptive_scurve_robust_v2 import prepare_context, apply_preprocess, make_post_context
from tools.preprocess_boundary_blend import feather_preprocess
from tools.doc_warp_composition import inverse_residual, source_support

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
CACHE = DATA/'adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020/evaluation/camera_docwarp_robust_20261021'
OLD = DATA/'camera_docquad_v1_pre_v2_1_20261021'
INVENTORY = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016/inventory.json')


def retain(source, target):
    """@luna-comment
    Input: Immutable file and new destination. Output: Byte-identical retained artifact.
    Key Steps: Copy JSON; hardlink immutable images or copy fallback.
    Key Params: Never overwrite existing targets. State/Dependencies: Own new output only.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if source.suffix == '.json':
        shutil.copyfile(source, target)
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)


def apply_common_delta(source, delta):
    """@luna-comment
    Input: Native uint8 BGR and scalar float correction at source resolution.
    Output: Gamut-safe BGR preserving exact channel differences.
    Key Steps: Round once; clamp to shared channel headroom; add common offset.
    Key Params: No independent channel clipping. State/Dependencies: No source mutation.
    """
    d = np.rint(delta).astype(np.int16)
    d = np.maximum(d, -source.min(axis=2).astype(np.int16))
    d = np.minimum(d, 255-source.max(axis=2).astype(np.int16))
    return (source.astype(np.int16)+d[:, :, None]).astype(np.uint8)


def process_frame(args):
    """@luna-comment
    Input: Original inventory row/new root and expected OCR model identity.
    Output: Original-sized PNG, corrected warp stages, contexts and bound receipt.
    Key Steps: Bind accepted repaired contour/quad/OCR/cache hashes; regenerate V2.2
    plus boundary blending; inverse-map only corrections; composite first-owned
    support in confidence order; validate untouched exterior/color and PNG roundtrip.
    Key Params: Repaired geometry unchanged, OCR12%/24px feather, document feather8px.
    State/Dependencies: Immutable corrected cache; no model inference or RGB double warp.
    """
    row, output, expected_model = args
    output = Path(output)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    source_path = Path(row['remote'])
    assert sha(source_path) == row['sha256']
    original = cv2.imread(str(source_path))
    directory = CACHE/'documents'/(row['id']+'.jpg')
    record_file = directory/'result.json'
    record = json.loads(record_file.read_text())
    assert list(original.shape[:2]) == [record['image_height'], record['image_width']]
    previous = json.loads((OLD/'documents'/row['name']/'result.json').read_text())
    old_instances = {v['instance_id']:v for v in previous['instances']}
    assert len(old_instances) == len(record['instances'])
    retain(source_path, output/'source'/row['name'])
    retain(record_file, output/'documents'/row['id']/'result.json')
    retain(directory/'overlay.png', output/'documents'/row['id']/'overlay.png')
    retain(OLD/'documents'/row['name']/'overlay.png', output/'geometry_before'/(row['id']+'.png'))
    preview(output/'previews'/(row['id']+'_source.jpg'), original)
    preview(output/'previews'/(row['id']+'_quad.jpg'), cv2.imread(str(directory/'overlay.png')))
    preview(output/'previews'/(row['id']+'_old_quad.jpg'), cv2.imread(str(OLD/'documents'/row['name']/'overlay.png')))
    owned = np.zeros(original.shape[:2], bool)
    correction = np.zeros(original.shape[:2], np.float32)
    stages = []
    for item in sorted(record['instances'], key=lambda v:(-v['confidence'],v['instance_id'])):
        prior = old_instances[item['instance_id']]
        assert np.array_equal(item['raw_contour_xy'],prior['raw_contour_xy'])
        assert abs(item['confidence']-prior['confidence']) < 1e-6
        changed_geometry = not np.array_equal(item['quad'],prior['quad'])
        if prior['geometry_type'] == 'boundary_quad':
            assert not changed_geometry
        key = row['id']+'_doc'+str(item['instance_id']).zfill(3)
        stem = Path(row['name']).stem+'__doc'+str(item['instance_id']).zfill(3)
        ocr_path = CACHE/'ocr'/row['id']/(str(item['instance_id']).zfill(3)+'.json')
        ocr = json.loads(ocr_path.read_text())
        warp_path = directory/item['image_file']
        assert ocr['source_sha256'] == row['sha256']
        assert ocr['warp_sha256'] == sha(warp_path)
        assert ocr['model_hashes'] == expected_model and ocr['fresh_warp_detection']
        assert np.array_equal(ocr['transform'],item['warp']['image_to_output'])
        warp = cv2.imread(str(warp_path))
        assert list(warp.shape[:2]) == ocr['warp_hw']
        valid_path = CACHE/ocr['valid_file']
        assert sha(valid_path) == ocr['valid_sha256']
        valid = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(valid,source_support(original.shape[:2],ocr['transform'],warp.shape[:2]))
        context = prepare_context(warp,ocr['boxes'],reference_id=row['sha256']+':'+key+':'+ocr['warp_sha256'])
        unblended = apply_preprocess(warp,context)
        processed, alpha = feather_preprocess(warp,unblended,ocr['boxes'])
        assert np.array_equal(processed[alpha == 0],warp[alpha == 0])
        assert np.array_equal(processed[alpha == 255],unblended[alpha == 255])
        delta, mapped_alpha, take, geometry = inverse_residual(original,warp,processed,ocr['transform'],item['quad'],valid,owned)
        assert np.array_equal(delta[:, :, 0],delta[:, :, 1]) and np.array_equal(delta[:, :, 1],delta[:, :, 2])
        correction += delta[:, :, 0]
        owned |= take
        warp_relative = 'warps/'+stem+'.png'
        pre_relative = 'warped_preprocessed/'+stem+'.png'
        context_relative = 'contexts/'+stem+'.context.json'
        retain(warp_path,output/warp_relative)
        retain(ocr_path,output/'ocr'/(key+'.json'))
        retain(valid_path,output/'ocr'/(key+'.valid.png'))
        overlay_path = CACHE/ocr['overlay']
        retain(overlay_path,output/'ocr'/(key+'.overlay.png'))
        write_image(output/pre_relative,processed)
        assert np.array_equal(cv2.imread(str(output/pre_relative)),processed)
        save_json(output/context_relative,make_post_context(context))
        preview(output/'previews'/(key+'_warp.jpg'),warp)
        preview(output/'previews'/(key+'_pre.jpg'),processed)
        preview(output/'previews'/(key+'_ocr.jpg'),cv2.imread(str(overlay_path)))
        stages.append(dict(key=key,instance_id=item['instance_id'],warp=warp_relative,pre=pre_relative,
                           context=context_relative,pre_sha256=sha(output/pre_relative),context_sha256=sha(output/context_relative),
                           warp_sha256=ocr['warp_sha256'],ocr_sha256=sha(ocr_path),changed_geometry=changed_geometry,
                           geometry_type=item['geometry_type'],boxes=len(ocr['boxes']),geometry=geometry,
                           reused_ocr_bound_to_repaired_warp=True,source_edge_exact=True,interior_pre_exact=True))
    final = apply_common_delta(original,correction)
    assert final.shape == original.shape and np.array_equal(final[~owned],original[~owned])
    for y0 in range(0,original.shape[0],128):
        change = final[y0:y0+128].astype(np.int16)-original[y0:y0+128].astype(np.int16)
        assert np.array_equal(change[:, :, 0],change[:, :, 1]) and np.array_equal(change[:, :, 1],change[:, :, 2])
    if not stages or not sum(s['boxes'] for s in stages):
        assert np.array_equal(final,original)
    target = output/'images'/(source_path.stem+'.png')
    write_image(target,final)
    assert np.array_equal(cv2.imread(str(target)),final)
    preview(output/'previews'/(row['id']+'_final.jpg'),final)
    write_image(output/'ownership'/(row['id']+'.png'),owned.astype(np.uint8)*255)
    receipt = dict(id=row['id'],name=row['name'],source_sha256=row['sha256'],shape=list(original.shape),
                   image=str(target.relative_to(output)),image_sha256=sha(target),
                   corrected_document_record_sha256=sha(record_file),instances=stages,
                   outside_support_unchanged=True,common_channel_delta=True,png_roundtrip_exact=True,
                   changed_pixels=int(np.count_nonzero(np.any(final != original,axis=2))),
                   status='processed' if stages else 'no_document_original_unchanged',seconds=time.perf_counter()-started)
    save_json(output/'receipts'/(row['id']+'.json'),receipt)
    return receipt


def slider(left, right, key):
    """@luna-comment
    Input: Relative preview URLs and unique ID. Output: Local-file compatible slider HTML.
    Key Steps: Layer left over right and bind clipping; Key Params: midpoint default.
    State/Dependencies: No external JavaScript or network requests.
    """
    return '<input type="range" min="0" max="100" value="50" style="width:90%" oninput="document.getElementById(\''+key+'\').style.clipPath=\'inset(0 \'+(100-this.value)+\'% 0 0)\'"><div class="pair"><img src="'+right+'"><img class="after" id="'+key+'" src="'+left+'"></div>'


def gallery(output, records):
    """@luna-comment
    Input: Completed original-frame and warp receipts. Output: Full-frame/stage gallery.
    Key Steps: Hash outputs; show corrected vs old quads, full-frame slider and per-warp
    OCR/pre views; validate all relative links. Key Params: native links expose full pixels.
    State/Dependencies: Own completed artifacts; no model-quality claims.
    """
    style = '<meta charset="utf-8"><style>body{font:17px sans-serif;background:#181c22;color:#eee;margin:25px}a{color:#8cd8ff}img{max-width:100%}.pair{position:relative;max-width:1200px}.pair img{display:block;width:100%}.after{position:absolute;top:0;left:0;clip-path:inset(0 50% 0 0)}</style>'
    for idx,r in enumerate(records):
        assert sha(output/r['image']) == r['image_sha256']
        body = style+'<h1>'+html.escape(r['id']+' | '+r['name'])+'</h1><a href="../index.html">All images</a>'
        for label,n in [('Previous',idx-1),('Next',idx+1)]:
            if 0 <= n < len(records): body += ' | <a href="'+records[n]['id']+'.html">'+label+'</a>'
        body += '<p>Accepted robust perspective geometry + latest V2.2 + OCR boundary blend + inverse correction on original pixels. No TextSR model/postprocessing.</p>'
        body += '<p>Original size: '+str(r['shape'][1])+' x '+str(r['shape'][0])+'; documents: '+str(len(r['instances']))+'; '+r['status']+'</p>'
        body += '<p><a href="../'+r['image']+'">Native original-sized result</a> | <a href="../source/'+r['name']+'">Original</a></p><h2>Full frame: left=result; right=original</h2>'
        body += slider('../previews/'+r['id']+'_final.jpg','../previews/'+r['id']+'_source.jpg','frame')
        body += '<h2>Geometry: left=corrected; right=old rectangular fallback</h2>'
        body += slider('../previews/'+r['id']+'_quad.jpg','../previews/'+r['id']+'_old_quad.jpg','geometry')
        body += '<p><a href="../documents/'+r['id']+'/overlay.png">Corrected native geometry</a></p>'
        for s in r['instances']:
            assert sha(output/s['pre']) == s['pre_sha256'] and sha(output/s['context']) == s['context_sha256']
            key=s['key']
            body += '<h2>'+key+' | geometry changed='+str(s['changed_geometry'])+' | OCR boxes='+str(s['boxes'])+'</h2>'
            body += '<p><a href="../'+s['warp']+'">Corrected warp</a> | <a href="../'+s['pre']+'">Preprocessed warp</a> | <a href="../'+s['context']+'">Warp post context</a></p>'
            body += '<img src="../previews/'+key+'_ocr.jpg"><p>Left=blended preprocessing; right=corrected source warp.</p>'
            body += slider('../previews/'+key+'_pre.jpg','../previews/'+key+'_warp.jpg',key)
        (output/'pages'/(r['id']+'.html')).write_text(body,encoding='utf-8')
    index=style+'<h1>Robust document geometry / V2.2 saturation + blend / original-size output</h1><p>Final model inputs: images/. Corrected document intermediates: warped_preprocessed/. Contexts apply to warp coordinates.</p><ul>'
    index+=''.join('<li><a href="pages/'+r['id']+'.html">'+html.escape(r['id']+' '+r['name'])+'</a></li>' for r in records)+'</ul>'
    (output/'index.html').write_text(index,encoding='utf-8')
    pages=[output/'index.html',*sorted((output/'pages').glob('*.html'))];links=0
    for page in pages:
        for link in re.findall(r'(?:href|src)="([^"]+)"',page.read_text()):
            assert (page.parent/html.unescape(link)).is_file(),(page,link)
            links+=1
    save_json(output/'gallery_audit.json',dict(status='passed',pages=len(pages),links=links))


def main():
    """@luna-comment
    Input: Fresh remote output and optional tilted/multi-document/color pilot.
    Output:280 native images plus307 corrected warp stages and verified summary.
    Key Steps: Audit repair metadata; execute bounded CPU workers; geometry counts,
    native identity and gallery checks before passed status.
    Key Params: Pilot013/045/072, four workers; immutable cached geometry/OCR reuse.
    State/Dependencies: Explicit accepted repaired cache, original source inventory.
    """
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name in ('images','pages','previews','receipts','contexts','warps','warped_preprocessed','ownership'):(out/name).mkdir()
    started=time.perf_counter()
    rows=json.loads(INVENTORY.read_text())['images']
    assert len(rows)==280
    proof=json.loads((CACHE/'GEOMETRY_VALIDATED.json').read_text())
    assert proof['passed'] and proof['revised_approximate_instances']==135 and proof['preserved_successful_boundary_instances']==155
    for name in ('GEOMETRY_VALIDATED.json','REPAIR_GEOMETRY.json','RUN_SOURCE_MANIFEST.json'):retain(CACHE/name,out/('upstream_'+name))
    expected=json.loads((INVENTORY.parent/'detections/IMG_20260605_011836092.json').read_text())['model_hashes']
    if args.pilot:rows=[r for r in rows if r['id'] in ('013','045','072')]
    save_json(out/'inventory.json',dict(images=rows,upstream_sha256=sha(INVENTORY)))
    save_json(out/'code_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')})
    try:
        records=[]
        save_json(out/'status.json',dict(status='running',completed=0,expected=len(rows)))
        with ProcessPoolExecutor(max_workers=4) as pool:
            for r in pool.map(process_frame,[(r,str(out),expected) for r in rows]):
                records.append(r);save_json(out/'status.json',dict(status='running',completed=len(records),expected=len(rows)))
                print('NATIVE',len(records),len(rows),r['id'],flush=True)
        stages=[s for r in records for s in r['instances']]
        revised=sum(s['changed_geometry'] for s in stages)
        if not args.pilot:assert len(stages)==307 and revised==135
        gallery(out,records)
        summary=dict(status='passed',frames=len(records),document_instances=len(stages),revised_geometry=revised,
                     ocr_boxes=sum(s['boxes'] for s in stages),original_size=True,model_inference=False,
                     geometry_cache=str(CACHE),no_document=[r['id'] for r in records if not r['instances']],
                     unchanged_frames=[r['id'] for r in records if not r['changed_pixels']],wall_seconds=time.perf_counter()-started,records=records)
        save_json(out/'summary.json',summary)
        save_json(out/'status.json',{k:v for k,v in summary.items() if k!='records'})
        print('PASS',len(records),'native frames',flush=True)
    except Exception as error:
        save_json(out/'status.json',dict(status='failed',error=repr(error)))
        raise


if __name__=='__main__':main()
