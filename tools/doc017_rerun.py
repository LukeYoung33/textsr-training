"""Isolated 017 geometry/OCR/native-preprocessing replay; preserve all prior outputs."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))
OUTPUT = BUILD.parent
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OLD = DATA/'camera_robust_pre_native_v22_20261021'
CACHE = OUTPUT/'geometry_cache'
UPSTREAM = DATA/'adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020/evaluation/camera_docwarp_robust_20261021'


def main():
    """@luna-comment
    Input: One phase and the accepted isolated geometry-code snapshot.
    Output: Fresh 017 warp/OCR/native result and hash-bound audit receipts.
    Key Steps: Refit unchanged contour, recompute homography, rerun OCR, then reuse
    released preprocessing/composition functions with only cache routing changed.
    Key Params: Warp caps8192/33554432; OCR and preprocessing keep released settings.
    State/Dependencies: Own fresh output only; original images/models remain read-only.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['geometry', 'ocr', 'native'])
    args = parser.parse_args()
    from tools.run_adaptive_pre_v21_camera_batch import sha, save_json
    row = next(r for r in json.loads((OLD/'inventory.json').read_text())['images'] if r['id']=='017')
    if args.phase == 'geometry':
        from tools.doc_quad_robust import repair_quad
        from tools import doc_quad_infer as base
        record = json.loads((UPSTREAM/'documents/017.jpg/result.json').read_text())
        original = cv2.imread(row['remote'])
        assert sha(row['remote']) == row['sha256']
        assert list(original.shape[:2]) == [record['image_height'],record['image_width']]
        target = CACHE/'documents/017.jpg'
        target.mkdir(parents=True, exist_ok=False)
        overlay = original.copy()
        expected = json.loads((OUTPUT/'expected_candidate.json').read_text())
        for i, previous in enumerate(record['instances']):
            item = repair_quad(previous['raw_contour_xy'],original.shape[:2],previous)
            assert np.allclose(item['quad'],expected['quad'],atol=1e-4)
            assert item['repair_evidence']['exterior_fallback_used']
            warp, metadata = base.warp_quad(original,item['quad'],max_side=8192,max_pixels=33554432,allow_outside=True)
            item['warp'] = metadata
            record['instances'][i] = item
            base.write_image(target/item['image_file'],warp)
            cv2.polylines(overlay,[np.rint(item['quad']).astype(np.int32)],True,(0,220,0),7,cv2.LINE_AA)
        base.write_image(target/'overlay.png',overlay)
        save_json(target/'result.json',record)
        save_json(OUTPUT/'geometry_audit.json',dict(source_sha256=row['sha256'],
            code_sha256=sha(BUILD/'tools/doc_quad_robust.py'),remote_matches_candidate=True,
            recomputed_warp=True,record_sha256=sha(target/'result.json'),instances=record['instances']))
    elif args.phase == 'ocr':
        from tools import evaluate_doc_warp_matrix as evaluation
        evaluation.OUT = CACHE
        evaluation.detect_warped_text(SimpleNamespace(rank=0,shards=1),[row])
    else:
        from tools import run_robust_pre_native as native
        native.CACHE = CACHE
        for name in ['images','pages','previews','receipts','contexts','warps','warped_preprocessed','ownership']:
            (OUTPUT/name).mkdir(exist_ok=True)
        expected = json.loads((native.INVENTORY.parent/'detections/IMG_20260605_011836092.json').read_text())['model_hashes']
        receipt = native.process_frame((row,str(OUTPUT),expected))
        native.gallery(OUTPUT,[receipt])
        for p in [OUTPUT/'pages/017.html',OUTPUT/'index.html']:
            s = p.read_text()
            note = '<div style="background:#fff0c2;padding:16px"><b>017 修正版：</b>保留原检测轮廓，允许画外四角估计，重新计算透视 warp、重新检测 OCR，并执行原 V2.2 预处理与逆映射回贴。画外像素仅为填充，未恢复真实内容。<a href="../../camera_robust_pre_native_v22_20261021/pages/017.html">查看原版017</a></div>'
            if p.name=='index.html': note=note.replace('../../camera_','../camera_')
            import re
            if re.search(r'<body[^>]*>',s):
                s=re.sub(r'(<body[^>]*>)',lambda m:m.group(1)+note,s,count=1)
            else:
                s=note+s
            p.write_text(s)
        save_json(OUTPUT/'summary.json',dict(status='passed',frames=1,records=[receipt],fresh_ocr=True))
    print('PHASE_PASSED',args.phase,flush=True)


if __name__ == '__main__':
    main()
