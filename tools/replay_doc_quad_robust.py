"""Replay retained model contours into a separate robust document pipeline root."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import doc_quad_infer as base
from tools.doc_quad_robust import repair_quad

OLD=ROOT.parent.parent/'camera_docwarp_20261021'
OUT=ROOT.parent
BASE=OLD.parent/'camera_final_20261021'


def retain_tree(source,destination):
    """@luna-comment
    Input: immutable completed file/tree and new isolated destination.
    Output: byte-identical copied JSON and hardlinked images, with copy fallback.
    Key Steps: recurse; never modify existing source or hardlink mutable JSON state.
    Key Params: only completed per-image artifacts, no progress/completion markers.
    State/Dependencies: same-filesystem hardlinks reduce redundant large PNG storage.
    """
    if source.is_dir():
        destination.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            retain_tree(path,destination/path.name)
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            return
        if source.suffix=='.json':
            shutil.copy2(source,destination)
        else:
            try:
                os.link(source,destination)
            except OSError:
                shutil.copy2(source,destination)


def replay_frame(path):
    """@luna-comment
    Input: one completed baseline document result JSON. Output: repaired per-frame record.
    Key Steps: robust-fit same contours; regenerate changed warp/overlay; preserve valid
    baseline fits and reuse downstream artifacts only for entirely unchanged geometry.
    Key Params:8192/33554432 allocation caps; approximate outside support remains explicit.
    State/Dependencies: original source and frozen previous outputs read-only.
    """
    cv2.setNumThreads(1)
    path=Path(path);record=json.loads(path.read_text())
    height,width=record['image_height'],record['image_width']
    changed=0
    for index,item in enumerate(record['instances']):
        result=repair_quad(item['raw_contour_xy'],(height,width),item)
        if result.get('repair_evidence',{}).get('changed'):
            record['instances'][index]=result;changed+=1
    destination=OUT/'documents'/path.parent.name
    destination.mkdir(parents=True,exist_ok=True)
    identity=path.parent.stem
    if not changed:
        retain_tree(path.parent,destination)
        for folder in ('ocr','prepared'):
            source=OLD/folder/identity
            if source.exists():
                retain_tree(source,OUT/folder/identity)
        for arm in ('01_v5_real','02_v3_real','03_v5_synthetic','04_v3_synthetic'):
            for folder,suffix in (('instances',''),('final','.png'),('ownership','.png'),('previews','.jpg'),('receipts','.json')):
                source=OLD/arm/folder/(identity+suffix)
                if source.exists():
                    retain_tree(source,OUT/arm/folder/(identity+suffix))
    else:
        # Detach previously reused artifacts before a new geometry revision.
        stale=OUT/'invalidated_cache'/str(time.time_ns())
        for folder in ('ocr','prepared'):
            cached=OUT/folder/identity
            if cached.exists():
                target=stale/folder/identity;target.parent.mkdir(parents=True,exist_ok=True)
                cached.rename(target)
        for arm in ('01_v5_real','02_v3_real','03_v5_synthetic','04_v3_synthetic'):
            for folder,suffix in (('instances',''),('final','.png'),('ownership','.png'),('previews','.jpg'),('receipts','.json')):
                cached=OUT/arm/folder/(identity+suffix)
                if cached.exists():
                    target=stale/arm/folder/cached.name;target.parent.mkdir(parents=True,exist_ok=True)
                    cached.rename(target)
        image=base.read_image(BASE/'source'/path.parent.name)
        overlay=image.copy()
        for item in record['instances']:
            warp,metadata=base.warp_quad(image,item['quad'],max_side=8192,max_pixels=33554432,allow_outside=item['geometry_type']=='approximate_quad')
            item['warp']=metadata
            artifact=destination/item['image_file']
            artifact.unlink(missing_ok=True)
            base.write_image(artifact,warp)
            cv2.polylines(overlay,[np.rint(item['quad']).astype(np.int32)],True,(40,220,40) if item['geometry_type']=='boundary_quad' else (255,220,0),7,cv2.LINE_AA)
        (destination/'overlay.png').unlink(missing_ok=True)
        base.write_image(destination/'overlay.png',overlay)
        (destination/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    return dict(id=identity,documents=len(record['instances']),changed_instances=changed,changed=bool(changed))


def main():
    """@luna-comment
    Input: completed v1 retained contours and staged robust adapter.
    Output: all280 new geometry records plus isolated repair summary and detector marker.
    Key Steps: six independent CPU workers; aggregate changed cases; retain baseline
    model provenance; publish completion only after all document records succeed.
    Key Params: no new model predictions or changed model weights.
    State/Dependencies: this root only; immutable prior pipeline is not overwritten.
    """
    paths=sorted((OLD/'documents').glob('*/result.json'))
    assert len(paths)==280
    rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for row in pool.map(replay_frame,paths):
            rows.append(row)
            print(json.dumps(row),flush=True)
    (OUT/'REPAIR_GEOMETRY.json').write_text(json.dumps(dict(images=len(rows),changed_frames=sum(r['changed'] for r in rows),
         changed_instances=sum(r['changed_instances'] for r in rows),frames=rows),indent=2)+'\n')
    marker=json.loads((OLD/'documents/run.json').read_text())
    marker['geometry_adapter']='robust_observed_line_consensus_perspective'
    (OUT/'documents/run.json').write_text(json.dumps(marker,indent=2)+'\n')


if __name__=='__main__':
    main()
