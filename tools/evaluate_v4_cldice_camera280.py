"""Isolated historical V4 clDice inference on the accepted Mojito280 geometry."""
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

OUT = DATA/'v4_cldice_camera280_20261030'
FIX = DATA/'camera_robust_pre_native_v22_017_fix_20261022/geometry_cache'
CONFIG = ROOT/'options/train_bucket_3/lenovo_v4_cp_strong_20260911/train_lenovo_v4_cldice_250k_20260911.yml'
CHECKPOINT = HISTORY/'lenovo_v4_cp_strong_20260911/lenovo_v4_cldice_250k_20260911/models/net_g_250000.pth'


def worker(rank, ids=None):
    """@luna-comment
    Input: shard0..3 or explicit pilot IDs; approved read-only source/model caches.
    Output: native PNGs, restored document images and hash-bound receipts.
    Key Steps: strictV4 loading, verify cache geometry, height96 line restore,
    full-box feather and inverse residual; assert exterior identity; safe resume.
    Key Params:96x480/416 line tiles,16px line and8px document feather; no tone change.
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
    from tools.run_mojito_v5_height96_20261016 import restore_height96
    from tools.doc_warp_composition import inverse_residual
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    options = yaml.safe_load(CONFIG.read_text())
    assert options['network_g']['type'] == 'BasicTextLineR4StudentY4UVV4'
    assert options['train']['cldice_weight'] == .02
    binding = dict(config=str(CONFIG), config_sha256=sha(CONFIG), checkpoint=str(CHECKPOINT),
                   checkpoint_sha256=sha(CHECKPOINT), runner_sha256=sha(__file__),
                   preprocessing='none', geometry='text_line_height96_bicubic',
                   helpers={name:sha(ROOT/name) for name in (
                       'tools/run_mojito_v5_height96_20261016.py', 'tools/doc_warp_composition.py',
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
            restored, details = restore_height96(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), ocr['boxes'], model, 'cuda')
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
    save_json(OUT/(f'PILOT.{rank}.json' if ids else f'COMPLETE.{rank}.json'), dict(binding=binding, count=len(rows)))


def publish():
    """@luna-comment
    Input:280 complete native outputs and old comparison manifest.
    Output: independent2x2 HTML, manifest and verification audit.
    Key Steps: verify source/model/final hashes and sizes; retain labeled old references;
    embed existing viewer without changing any previous report.
    Key Params:280 rows, LQ/V4/V3raw/V5raw defaults; no GT metrics or quality judgments.
    State/Dependencies: Pillow and shared renderer; read-only old artifacts.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import render_html, file_url
    prior = json.loads((DATA/'camera_compare_2x2_20261022/manifest.json').read_text())
    inventory = {r['id']:r for r in json.loads(INV.read_text())['images']}
    assert len(prior['rows']) == len(inventory) == 280
    model = dict(key='v4_cldice', label='V4 · clDice · 09-11 · 250K',
                 note='文档warp → 文本行归一高96 → 96×480推理 → 边界融合 → unwarp；无亮度预处理')
    manifest = dict(title='Mojito280 · V4 clDice', subtitle='V4 clDice 与原图 / 历史 V3、V5 参考',
                    notes='V4：09-11 CP加强版，clDice=0.02，250K。017使用修复后的文档几何。\nV3/V5为历史参考，训练数据与损失不同，不是单一变量对照。',
                    defaults='lq,v4_cldice,original_v3_raw,original_v5_raw',
                    models=[m for m in prior['models'] if m['key'] in ('lq','original_v3_raw','original_v5_raw')]+[model], rows=[])
    entries = []
    binding = None
    for row in prior['rows']:
        identity = row['id']
        receipt = json.loads((OUT/'receipts'/(identity+'.json')).read_text())
        assert receipt['source_sha256'] == inventory[identity]['sha256']
        assert receipt['outside_owned_unchanged']
        if binding is None:
            binding = receipt['binding']
        assert receipt['binding'] == binding
        cache = FIX if identity == '017' else DOC
        source_name = identity+Path(inventory[identity]['name']).suffix.lower()
        assert receipt['document_record_sha256'] == sha(cache/'documents'/source_name/'result.json')
        final = OUT/receipt['final']
        assert sha(final) == receipt['final_sha256']
        with Image.open(final) as image:
            assert image.size == (row['width'], row['height'])
            assert list(reversed(image.size)) == receipt['source_hw']
            image.verify()
        output = {k:v for k,v in row.items() if k != 'images'}
        output['images'] = {k:v for k,v in row['images'].items() if k in ('original_v3_raw','original_v5_raw')}
        output['images']['v4_cldice'] = file_url(final)
        manifest['rows'].append(output)
        entries.append(dict(id=identity, final_sha256=receipt['final_sha256'], stages=len(receipt['stages'])))
    assert binding['checkpoint_sha256'] == sha(CHECKPOINT)
    save_json(OUT/'manifest.json', manifest)
    (OUT/'index.html').write_text(render_html(manifest), encoding='utf-8')
    save_json(OUT/'AUDIT.json', dict(status='passed', count=len(entries), binding=binding, entries=entries))


def execute(rank):
    """@luna-comment Input: GPU/shard0..3. Output: child exit status.
    Key Steps: log isolated worker and wait; preserve other GPU jobs.
    Key Params: one process per GPU, two CPU threads. State/Dependencies: subprocess/CUDA.
    """
    log = OUT/'logs'/f'worker.{rank}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as stream:
        result = subprocess.run([sys.executable, '-u', __file__, 'worker', '--rank', str(rank)],
                                stdout=stream, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
    return dict(rank=rank, exit_code=result.returncode)


def run():
    """@luna-comment Input: authorized280 batch. Output: four verified shards and viewer.
    Key Steps: exclusive lock, launch bounded pool, assert exits then publish.
    Key Params: fourGPUs; failure recorded without publishing incomplete gallery.
    State/Dependencies: Linux flock; detached parent; owned outputs only.
    """
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(OUT/'STATUS.json', dict(phase='running', pid=os.getpid(), total=280))
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(execute, range(4)))
            save_json(OUT/'WORKERS.json', results)
            assert all(r['exit_code'] == 0 for r in results), results
            publish()
            save_json(OUT/'STATUS.json', dict(phase='complete', total=280))
        except Exception as error:
            save_json(OUT/'STATUS.json', dict(phase='failed', error=str(error)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','worker','publish'))
    parser.add_argument('--rank', type=int, choices=range(4), default=0)
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args()
    if args.action == 'run':
        run()
    elif args.action == 'publish':
        publish()
    else:
        worker(args.rank, args.ids)
