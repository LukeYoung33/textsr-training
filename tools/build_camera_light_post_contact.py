"""Create source-coordinate 2x inspection sheets from the exact V5 baseline and three new results."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_camera_light_post_v2 import FIX, OUT, digest, mapped_path

MANIFEST = ROOT/'output/camera_2x2_comparison_20261022/manifest.json'


def sheet(identity):
    """@luna-comment Input: completed 280-list image ID. Output: two Y contact sheets with identical native crops in four columns. Key Steps: select largest/smallest OCR-height boxes, inverse-map centers, crop all exact finals and nearest-neighbor display2x. Key Params: 520x260 source ROI. State/Dependencies: read-only baseline/results; writes approved Y output root."""
    rows = {v['id']: v for v in json.loads(MANIFEST.read_text(encoding='utf-8'))['rows']}
    row = rows[identity]
    baseline_path = mapped_path(row['images']['vosr_v5_block'])
    receipt = json.loads((baseline_path.parent.parent/'receipts'/(identity+'.json')).read_text(encoding='utf-8'))
    record_path = FIX/'geometry_cache/documents/017.jpg/result.json' if identity == '017' else mapped_path(receipt['document_record'])
    record = json.loads(record_path.read_text(encoding='utf-8'))
    doc_root = FIX/'geometry_cache' if identity == '017' else record_path.parents[2]
    item = sorted(record['instances'], key=lambda v: (-v['confidence'], v['instance_id']))[0]
    ocr = json.loads((doc_root/'ocr'/identity/(f"{item['instance_id']:03d}.json")).read_text(encoding='utf-8'))
    boxes = sorted(ocr['boxes'], key=lambda b: b[3]-b[1])
    source_paths = [mapped_path(row['images']['vosr_v5_block'])]+[OUT/key/(identity+'.png') for key in ('light_v1_standard', 'light_v2_low', 'light_v2_standard')]
    labels = ['V5 baseline', 'V1 standard', 'V2 low', 'V2 standard']
    images = []
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(Image.open(path).convert('RGB'))
    transform = cv2.invert(np.asarray(item['warp']['image_to_output'], np.float64))[1]
    outputs = []
    for name, box in [('large_text', boxes[-1]), ('small_text', boxes[0])]:
        center = np.array([[[.5*(box[0]+box[2]), .5*(box[1]+box[3])]]], np.float64)
        center = cv2.perspectiveTransform(center, transform)[0, 0]
        x0 = max(0, min(row['width']-520, round(float(center[0]))-260))
        y0 = max(0, min(row['height']-260, round(float(center[1]))-130))
        crop = (x0, y0, x0+520, y0+260)
        board = Image.new('RGB', (4*1040, 570), (30, 35, 42))
        draw = ImageDraw.Draw(board)
        for index, (image, label) in enumerate(zip(images, labels)):
            display = image.crop(crop).resize((1040, 520), Image.Resampling.NEAREST)
            board.paste(display, (index*1040, 50))
            draw.text((index*1040+15, 16), label, fill='white')
        target = OUT/'inspection'/(identity+'_'+name+'_2x.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        board.save(target)
        outputs.append({'path': str(target), 'crop_xyxy': crop, 'ocr_box': box, 'sha256': digest(target)})
    (OUT/'inspection'/(identity+'.json')).write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    return outputs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('id')
    print(json.dumps(sheet(parser.parse_args().id), indent=2))
