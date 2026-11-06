"""Portable original-contour, old/new geometry and rectification comparisons."""
from concurrent.futures import ProcessPoolExecutor
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.doc_warp_gallery import embedded,figure,shell
from tools.replay_doc_quad_robust import OLD,OUT,BASE


def render_case(row):
    """@luna-comment
    Input: replay case ID/change metadata. Output: embedded old/new quad and warp page.
    Key Steps: draw identical raw contours and old/new geometry at preview resolution;
    pair native warps, expose approximation/support/coverage and per-case navigation.
    Key Params: display overlays only; no geometry or native image modifications.
    State/Dependencies: immutable old results and completed corrected geometry tree.
    """
    cv2.setNumThreads(1)
    identity=row['id'];old_path=OLD/'documents'/(identity+'.jpg');new_path=OUT/'documents'/(identity+'.jpg')
    old=json.loads((old_path/'result.json').read_text());new=json.loads((new_path/'result.json').read_text())
    source=cv2.imread(str(BASE/'previews'/(identity+'_source.jpg')))
    scale=[source.shape[1]/old['image_width'],source.shape[0]/old['image_height']]
    images=[]
    for record,label in ((old,'before'),(new,'after')):
        image=source.copy()
        for item in record['instances']:
            cv2.polylines(image,[np.rint(np.asarray(item['raw_contour_xy'])*scale).astype(np.int32)],True,(220,0,220),1)
            cv2.polylines(image,[np.rint(np.asarray(item['quad'])*scale).astype(np.int32)],True,(255,210,0) if label=='before' else (30,210,30),3,cv2.LINE_AA)
        path=OUT/'geometry/previews'/(identity+'_'+label+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(path),image);images.append(path)
    body=f'<h1>{identity} · Geometry repair</h1><nav><a href="../index.html">All cases</a><a href="{max(1,int(identity)-1):03d}.html">Previous</a><a href="{min(280,int(identity)+1):03d}.html">Next</a><a href="../../pages/{identity}.html">Four-model final comparison</a></nav><p>Magenta: unchanged model contour. Cyan: old quad. Green: repaired/preserved quad. Approximate geometry is not physical-corner ground truth.</p><section><div class="grid">'
    body+=figure('Before',images[0],'../../../camera_docwarp_20261021/documents/'+old_path.name+'/overlay.png')
    body+=figure('After',images[1],'../../documents/'+new_path.name+'/overlay.png')+'</div></section>'
    for old_item,item in zip(old['instances'],new['instances']):
        body+='<section><h2>Instance '+str(item['instance_id'])+'</h2><div class="grid">'
        body+=figure('Previous warp',old_path/old_item['image_file'],'../../../camera_docwarp_20261021/documents/'+old_path.name+'/'+old_item['image_file'])
        body+=figure('Repaired/preserved warp',new_path/item['image_file'],'../../documents/'+new_path.name+'/'+item['image_file'])+'</div>'
        body+='<pre>'+html.escape(json.dumps(dict(reason=item['selection_reason'],geometry_type=item['geometry_type'],
                  changed=item.get('repair_evidence',{}).get('changed',False),quad=item['quad'],
                  support=item['geometry_evidence'].get('robust_edge_consensus'),coverage=item['geometry_evidence'].get('mask_coverage')),indent=2))+'</pre></section>'
    (OUT/'geometry/pages'/(identity+'.html')).write_text(shell(identity+' geometry repair',body),encoding='utf-8')
    return identity


def main():
    """@luna-comment
    Input: completed280-case geometry replay. Output: linked offline comparison gallery.
    Key Steps: six independent page renderers; list changed and preserved source cases.
    Key Params:013 highlighted; no enhanced images required for geometry review.
    State/Dependencies: own geometry directory only; embedded previews/native links.
    """
    summary=json.loads((OUT/'REPAIR_GEOMETRY.json').read_text())
    (OUT/'geometry/pages').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for identity in pool.map(render_case,summary['frames']):
            print('GEOMETRY_PAGE',identity,flush=True)
    body='<h1>Document geometry repair · Same model contours</h1><p><a href="pages/013.html">Open highlighted013 case</a></p><p>'+str(summary['changed_frames'])+' changed frames / '+str(summary['changed_instances'])+' revised instances. Successful baseline boundary fits remain unchanged.</p><table><tr><th>Case</th><th>Changed instances</th></tr>'
    for row in summary['frames']:
        body+=f'<tr><td><a href="pages/{row["id"]}.html">{row["id"]}</a></td><td>{row["changed_instances"]}</td></tr>'
    (OUT/'geometry/index.html').write_text(shell('Geometry repair comparisons',body+'</table>'),encoding='utf-8')


if __name__=='__main__':
    main()
