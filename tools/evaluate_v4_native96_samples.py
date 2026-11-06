"""Four native96x480 full-warp V4 clDice samples versus height-normalized lines."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_camera_compare_missing import DATA, OLD, DOC, INV, HISTORY
from tools.evaluate_offline_camera_matrix import sha, save_json, image_write

OUT = DATA/'v4_native96_samples_20261030'
FIX = DATA/'camera_robust_pre_native_v22_017_fix_20261022/geometry_cache'
CONFIG = ROOT/'options/train_bucket_3/lenovo_v4_cp_strong_20260911/train_lenovo_v4_cldice_250k_20260911.yml'
CHECKPOINT = HISTORY/'lenovo_v4_cp_strong_20260911/lenovo_v4_cldice_250k_20260911/models/net_g_250000.pth'


def worker(rank=0, ids=('049', '050', '051', '052')):
    """@luna-comment
    Input: fixed049-052 sample IDs; approved read-only source/model caches.
    Output: native PNGs, restored document images and hash-bound receipts.
    Key Steps: strictV4 loading, verify cache geometry, native full-warp tiling,
    overlap fusion and inverse residual; assert exterior identity; safe resume.
    Key Params:96x480 tiles/64x448 stride,16px overlap and8px document feather; no resize or OCR gating.
    State/Dependencies: CUDA, existing helpers, isolated per-shard progress only.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    import importlib
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch')
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore
    from tools.doc_warp_composition import inverse_residual
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    options = yaml.safe_load(CONFIG.read_text())
    assert options['network_g']['type'] == 'BasicTextLineR4StudentY4UVV4'
    assert options['train']['cldice_weight'] == .02
    binding = dict(config=str(CONFIG), config_sha256=sha(CONFIG), checkpoint=str(CHECKPOINT),
                   checkpoint_sha256=sha(CHECKPOINT), runner_sha256=sha(__file__),
                   preprocessing='none', geometry='full_warp_native96x480',
                   helpers={name:sha(ROOT/name) for name in (
                       'tools/auto_hx_v3_native.py', 'tools/doc_warp_composition.py',
                       'realesrgan/archs/basic_text_line_r4_student_y4uv_v4_arch.py')})
    model = build_network(options['network_g']).cuda().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu')['params'], strict=True)
    inventory = json.loads(INV.read_text())['images']
    assert len(inventory) == 280
    rows = [r for r in inventory if r['id'] in ids] if ids else inventory[rank::4]
    if ids:
        assert len(rows) == len(ids)
    for done, row in enumerate(rows, 1):
        started = time.monotonic()
        identity = row['id']
        receipt_path = OUT/'receipts'/(identity+'.json')
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            assert old['binding'] == binding and old['source_sha256'] == row['sha256']
            assert sha(OUT/old['final']) == old['final_sha256']
            continue
        source_file = OLD/'evaluation/camera_final_20261021/source'/(identity+Path(row['name']).suffix.lower())
        assert sha(source_file) == row['sha256']
        source = cv2.imread(str(source_file))
        assert source is not None
        cache = FIX if identity == '017' else DOC
        record_file = cache/'documents'/source_file.name/'result.json'
        record = json.loads(record_file.read_text())
        assert [record['image_height'], record['image_width']] == list(source.shape[:2])
        canvas, owned = source.astype(np.float32), np.zeros(source.shape[:2], bool)
        stages = []
        for item in sorted(record['instances'], key=lambda x:(-x['confidence'], x['instance_id'])):
            number = item['instance_id']
            ocr_file = cache/'ocr'/identity/f'{number:03d}.json'
            ocr = json.loads(ocr_file.read_text())
            assert ocr['source_sha256'] == row['sha256']
            assert sha(ocr['warp_file']) == ocr['warp_sha256']
            assert sha(cache/ocr['valid_file']) == ocr['valid_sha256']
            assert np.allclose(ocr['transform'], item['warp']['image_to_output'])
            original = cv2.imread(ocr['warp_file'])
            height, width = original.shape[:2]
            restored, details = restore(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), [[0, 0, width, height]], model, 'cuda', box_feather=0)
            details.update(ocr_cropping=False, resized=False, full_warp_coverage=True)
            restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
            intermediate = OUT/'warps'/identity/f'{number:03d}.png'
            image_write(intermediate, restored)
            valid = cv2.imread(str(cache/ocr['valid_file']), 0)
            delta, _, take, geometry = inverse_residual(source, original, restored, item['warp']['image_to_output'], item['quad'], valid, owned)
            canvas += delta
            owned |= take
            stages.append(dict(instance_id=number, ocr_sha256=sha(ocr_file), warp_sha256=ocr['warp_sha256'],
                               restoration=details, geometry=geometry, restored_warp=str(intermediate),
                               restored_warp_sha256=sha(intermediate)))
        final = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
        assert np.array_equal(final[~owned], source[~owned])
        output = OUT/'final'/(identity+'.png')
        image_write(output, final)
        save_json(receipt_path, dict(id=identity, name=row['name'], binding=binding,
                  source_sha256=row['sha256'], source_hw=list(source.shape[:2]), final=str(output.relative_to(OUT)),
                  final_sha256=sha(output), document_record_sha256=sha(record_file), document_cache=str(cache),
                  stages=stages, outside_owned_unchanged=True, seconds=time.monotonic()-started))
        save_json(OUT/f'progress.{rank}.json', dict(done=done, total=len(rows), id=identity))
        print(json.dumps(dict(id=identity, seconds=round(time.monotonic()-started, 2))), flush=True)
    save_json(OUT/f'COMPLETE.{rank}.json', dict(binding=binding, count=len(rows)))


def publish():
    """@luna-comment
    Input: four completed native samples and previous280 receipts/viewer.
    Output: independent four-panel comparison and hash/geometry verification audit.
    Key Steps: match checkpoint/config/source/document/warp identities, verify outputs;
    expose LQ, height96, native96 and labeled historicalV3 reference.
    Key Params:049-052 only, native dimensions and full-warp coverage required.
    State/Dependencies: PIL/shared renderer; old outputs remain read-only.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import render_html, file_url
    previous = DATA/'v4_cldice_camera280_20261030'
    manifest = json.loads((previous/'manifest.json').read_text())
    manifest['rows'] = [r for r in manifest['rows'] if r['id'] in ('049','050','051','052')]
    assert len(manifest['rows']) == 4
    manifest['models'] = [m for m in manifest['models'] if m['key'] != 'original_v5_raw']
    manifest['models'].append(dict(key='v4_native96', label='V4 clDice · 原生96×480整幅切块',
                                  note='相同文档warp → 原生96×480 / 步长64×448 → 重叠融合 → unwarp；不按OCR筛选、不缩放文字'))
    manifest.update(title='V4 clDice · 049–052 · 文本行与原生切块',
                    subtitle='同权重：整行归一高96 vs 整幅warp原生96×480切块',
                    defaults='lq,v4_cldice,v4_native96,original_v3_raw',
                    notes='两组V4使用相同权重和文档几何，无亮度预处理。原生切块处理整个warp，包含背景；文本行版按OCR框处理并归一到高96。V3仅为历史参考。')
    entries = []
    for row in manifest['rows']:
        identity = row['id']
        record = json.loads((OUT/'receipts'/(identity+'.json')).read_text())
        baseline = json.loads((previous/'receipts'/(identity+'.json')).read_text())
        for key in ('source_sha256','source_hw','document_record_sha256'):
            assert record[key] == baseline[key]
        for key in ('checkpoint_sha256','config_sha256'):
            assert record['binding'][key] == baseline['binding'][key]
        assert record['outside_owned_unchanged'] and record['stages']
        assert len(record['stages']) == len(baseline['stages'])
        for stage, old in zip(record['stages'], baseline['stages']):
            assert stage['warp_sha256'] == old['warp_sha256']
            details = stage['restoration']
            assert details['canvas_hw'] == [96,480] and details['stride_hw'] == [64,448]
            assert details['full_warp_coverage'] and not details['resized'] and not details['ocr_cropping']
            assert details['box_feather'] == 0
        image = OUT/record['final']
        assert sha(image) == record['final_sha256']
        with Image.open(image) as pixels:
            assert pixels.size == (row['width'],row['height'])
            pixels.verify()
        row['images']['v4_native96'] = file_url(image)
        entries.append(dict(id=identity, final_sha256=record['final_sha256'],
                            patches=sum(s['restoration']['patches'] for s in record['stages'])))
    save_json(OUT/'manifest.json', manifest)
    (OUT/'index.html').write_text(render_html(manifest), encoding='utf-8')
    save_json(OUT/'AUDIT.json', dict(status='passed', count=4, entries=entries))


if __name__ == '__main__':
    worker()
    publish()

