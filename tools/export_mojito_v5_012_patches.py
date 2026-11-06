"""Capture actual V5 forward outputs for camera image012 without changing fusion."""
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    """@luna-comment
    Input: completed V5 image012 receipt, original/preprocessed images and checkpoint.
    Output: every native input/output/difference patch, raw float tensors, index and audit.
    Key Steps: strict load, hook actual network calls in unchanged shared restore,
    save ordered patches, verify reconstructed final is byte-identical to old result.
    Key Params: actual96x480/stride64x448/batch8; RGB display uses original clamp/round.
    State/Dependencies: isolated diagnostic output, read-only existing models/results.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch')
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore, positions, sha, save_json
    from tools.run_mojito_camera_v5_strict_h96_20261016 import BASE, OUT, TRAIN, CHECKPOINT, EXPECTED_SHA
    from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    name = 'IMG_20260605_014719726'
    receipt = json.loads((OUT/'receipts'/(name+'.json')).read_text())
    source = Path(receipt['source'])
    assert sha(source) == receipt['source_sha256'] and sha(CHECKPOINT) == EXPECTED_SHA
    original = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
    processed = cv2.cvtColor(cv2.imread(str(BASE/'preprocessed'/(name+'.png'))), cv2.COLOR_BGR2RGB)
    import hashlib
    assert hashlib.sha256(processed.tobytes()).hexdigest() == receipt['model_input_pixel_sha256']
    target = OUT/'diagnostic_012'
    for folder in ('patches', 'raw_y4uv', 'raw_rgb'):
        (target/folder).mkdir(parents=True, exist_ok=True)
    h, w = original.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    for x0,y0,x1,y1 in receipt['text_boxes']:mask[y0:y1,x0:x1] = 1
    grid = [(x,y) for y in positions(h,96,64) for x in positions(w,480,448)
            if mask[y:min(h,y+96),x:min(w,x+480)].any()]
    assert len(grid) == receipt['patches']
    config = yaml.safe_load((TRAIN/'configs/train_strict_combined_h96_250k_20261014.yml').read_text())
    network = build_network(config['network_g']).cuda().eval()
    network.load_state_dict(torch.load(CHECKPOINT,map_location='cpu')['params'],strict=True)
    rows = []

    def capture(module, inputs, prediction):
        """@luna-comment
        Input: actual forward batch and unclipped Y4UV output from shared restore.
        Output: ordered unblended native PNGs and raw float arrays, metadata appended.
        Key Steps: same training RGB decode, locate grid batch, save original and
        preprocessed crop plus output/difference and clipping statistics.
        Key Params: diff5x centered128, output RGB clipped0..1 then rounded to uint8.
        State/Dependencies: read-only grid/source images, rows cursor, diagnostic folder.
        """
        raw = YUV2RGB(y4uv_to_yuv444(prediction), is_train=True)
        assert torch.isfinite(raw).all()
        y4uv = prediction.detach().cpu().numpy()
        rgb = raw.detach().permute(0,2,3,1).cpu().numpy()
        start = len(rows)
        for offset, value in enumerate(rgb):
            i = start+offset
            x,y = grid[i]
            crops = []
            for image in (original, processed):
                crop = image[y:y+96,x:x+480]
                crops.append(np.pad(crop,((0,96-crop.shape[0]),(0,480-crop.shape[1]),(0,0)),mode='edge'))
            output = np.uint8(np.clip(np.rint(value.clip(0,1)*255),0,255))
            difference = np.uint8(np.clip(128+5*(output.astype(np.float32)-crops[1]),0,255))
            tag = f'{i+1:04d}'
            for key, pixels in [('source',crops[0]),('input',crops[1]),('output',output),('diff',difference)]:
                assert cv2.imwrite(str(target/'patches'/(tag+'_'+key+'.png')),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
            np.save(target/'raw_y4uv'/(tag+'.npy'),y4uv[offset])
            np.save(target/'raw_rgb'/(tag+'.npy'),value)
            rows.append(dict(id=tag,x=x,y=y,width=480,height=96,
                valid_width=min(480,w-x),valid_height=min(96,h-y),
                mae_vs_input=float(np.abs(output.astype(float)-crops[1]).mean()),
                rgb_min=float(value.min()),rgb_max=float(value.max()),
                clipped_fraction=float(((value<0)|(value>1)).mean())))
    hook = network.register_forward_hook(capture)
    result, details = restore(processed,receipt['text_boxes'],network,'cuda',blend_source=original)
    hook.remove()
    old = cv2.cvtColor(cv2.imread(receipt['output']),cv2.COLOR_BGR2RGB)
    assert np.array_equal(result,old), 'Diagnostic changed final output'
    assert len(rows) == len(grid)
    save_json(target/'index.json',dict(status='passed',image_id='012',name=source.name,
        source_sha256=receipt['source_sha256'],checkpoint_sha256=EXPECTED_SHA,
        model_input_pixel_sha256=receipt['model_input_pixel_sha256'],
        final_pixel_identical=True,patches=len(rows),width=w,height=h,
        text_boxes=receipt['text_boxes'],details=details,rows=rows))
    print(json.dumps(dict(passed=True,patches=len(rows),final_pixel_identical=True,output=str(target))))


if __name__ == '__main__':
    main()
