"""Run raw-camera OCR, historical V5 preprocessing,0928 H96 and full postprocess."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_0928_full_pipeline_20261019')
TRAIN = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_8arm_20260928/lenovo_v5_r1_bglf_uv_v5_borderrail_h96_250k_20260928')
CHECKPOINT = TRAIN/'models/net_g_250000.pth'
CONFIG = TRAIN/'train_v5_borderrail_h96_250k_20260928.yml'
EXPECTED_SHA = '7c1b1efcbd231a407eeb106c19fcdeb6051ec500938c9ea29d40dc64e77171eb'
MODELS = Path('/home/guang.yang/workspace/PaddleOCR-3.7.0/deploy/ppocrv6_tiny_onnx/paddle_models')


def sha(path):
    """@luna-comment Input:file path. Output:SHA256. Key Steps:stream1MiB blocks.
    Key Params:encoded bytes. State/Dependencies:read-only asset, no mutations.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):digest.update(block)
    return digest.hexdigest()


def save(path, value):
    """@luna-comment Input:finite JSON object/path. Output:atomic UTF8 receipt.
    Key Steps:sibling temporary then replace. Key Params:reject NaN.
    State/Dependencies:isolated task output directories only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp');tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8');tmp.replace(path)


def snapshots(names):
    """@luna-comment
    Input:ordered historical module names and task snapshot manifest.
    Output:verified imported modules under process-local canonical tools names.
    Key Steps:verify source bytes, register module before execution, load unchanged code.
    Key Params:explicit allowlisted manifest; dependency order supplied by adapter.
    State/Dependencies:isolated Python process module table, no repository overwrite.
    """
    manifest = json.loads((OUT/'code/manifest.json').read_text())
    loaded = {}
    for name in names:
        path = OUT/'code'/(name+'.py')
        assert sha(path) == manifest[name+'.py']
        spec = importlib.util.spec_from_file_location('tools.'+name, path)
        module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module
        spec.loader.exec_module(module);loaded[name] = module
    return loaded


def prepare(rank, shards, limit):
    """@luna-comment
    Input:exact current raw originals, frozen tiny det/rec models and historical code.
    Output:new normalized OCR, selected V5 borderrail full-frame PNGs and receipts.
    Key Steps:run original OCR settings; original normalize/crop/tiled preprocessing;
    retain selected arm and call original full-box8px composer without algorithm edits.
    Key Params:OCR1600/.3/.6/1.5,rec threshold0; bounded4CPU threads; no prior geometry.
    State/Dependencies:Paddle environment, read-only staged raw images, own outputs.
    """
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR
    modules = snapshots(['text_lq_scurve_mild', 'materialize_lenovo_mild_scurve_lines',
                         'text_gt_fitted_v5_v6_pair', 'materialize_lenovo_v5_v6_ocr_matrix',
                         'run_lenovo_mojito_test_ocr', 'run_lenovo_mojito_test_v5_v6'])
    normalize = modules['run_lenovo_mojito_test_ocr']._normalize
    legacy = modules['run_lenovo_mojito_test_v5_v6']
    cv2.setNumThreads(1)
    detector = MODELS/'PP-OCRv6_tiny_det_infer';recognizer = MODELS/'PP-OCRv6_tiny_rec_infer'
    model_hashes = {str(p.relative_to(MODELS)): sha(p) for folder in [detector, recognizer] for p in folder.iterdir() if p.is_file()}
    engine = PaddleOCR(text_detection_model_name='PP-OCRv6_tiny_det', text_detection_model_dir=str(detector),
                      text_recognition_model_name='PP-OCRv6_tiny_rec', text_recognition_model_dir=str(recognizer),
                      use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                      text_det_limit_side_len=1600, text_det_limit_type='max', text_det_thresh=.3,
                      text_det_box_thresh=.6, text_det_unclip_ratio=1.5, text_rec_score_thresh=0.,
                      text_recognition_batch_size=1, return_word_box=True, ocr_version='PP-OCRv6', lang='ch',
                      device='cpu', cpu_threads=4)
    rows = json.loads((OUT/'inventory.json').read_text())['images'][:limit]
    for row in rows[rank::shards]:
        started = time.perf_counter();stem = Path(row['name']).stem
        ready = OUT/'prepared'/(stem+'.json')
        if ready.exists():
            old = json.loads(ready.read_text());assert old['source_sha256'] == row['sha256']
            assert sha(old['preprocessed']) == old['preprocessed_sha256'];continue
        source_path = Path(row['remote']);assert sha(source_path) == row['sha256']
        raw = cv2.imread(str(source_path));assert raw is not None
        result = list(engine.predict(input=raw));assert len(result) == 1
        payload = result[0].json
        if isinstance(payload, str):payload = json.loads(payload)
        normalized = normalize(payload, raw.shape[:2])
        normalized.update(source=str(source_path), source_sha256=row['sha256'], source_hw=list(raw.shape[:2]), ocr_model='PP-OCRv6_tiny')
        ocr_path = OUT/'ocr'/(stem+'.json');save(ocr_path, normalized)
        save(OUT/'raw_ocr'/(stem+'.json'), payload)
        regions = []
        for index, annotation in enumerate(normalized['ocr']['annotations']):
            box = legacy._box(annotation, raw.shape[:2])
            if box is None:continue
            x0, y0, x1, y1 = box
            processed, telemetry = legacy._process_tiled(raw[y0:y1, x0:x1])
            line_path = OUT/'lines'/row['id']/f'{index:03d}.png';line_path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(line_path), processed['v5_borderrail'], [cv2.IMWRITE_PNG_COMPRESSION, 3])
            regions.append(dict(ocr_index=index, ocr_box=list(box), files=dict(v5_borderrail=str(line_path)), tiles=telemetry))
        image, outside_changed = legacy._compose(raw, regions, 'v5_borderrail', 8)
        path = OUT/'preprocessed'/(stem+'.png');path.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        save(ready, dict(id=row['id'], name=row['name'], source=str(source_path), source_sha256=row['sha256'],
                        width=raw.shape[1], height=raw.shape[0], ocr=str(ocr_path), ocr_sha256=sha(ocr_path),
                        preprocessed=str(path), preprocessed_sha256=sha(path), regions=regions,
                        outside_preprocess_changed=outside_changed, ocr_model_hashes=model_hashes,
                        seconds=time.perf_counter()-started))
        print('PREPARED', row['id'], len(regions), round(time.perf_counter()-started, 2), flush=True)


def infer(rank, shards, limit):
    """@luna-comment
    Input:new historical OCR/preprocessed receipt and pinned0928 main250K checkpoint.
    Output:native model/final PNGs, original inference traces and full post receipts.
    Key Steps:strict original builder, original runner/default normal inference,
    full historical apply_postprocess, dimension/finite/hash checks and previews.
    Key Params:p17height96,overlap64,blend8,context.12/.03; post conservative/no filter.
    State/Dependencies:CUDA process, checked immutable prepared inputs, no new warping.
    """
    import importlib
    import cv2
    import numpy as np
    import torch
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch')
    post = snapshots(['text_lq_scurve_mild', 'postprocess_lenovo_borderrail_normal_v5_20260929'])['postprocess_lenovo_borderrail_normal_v5_20260929']
    from tools.basic_v2_eval_network import build_eval_network
    from tools.infer_basic_text_line_full_image import NativeY4UVTileRunner, infer_full_image, load_lq_preprocess_config
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert sha(CHECKPOINT) == EXPECTED_SHA
    network, digest = build_eval_network(CHECKPOINT, torch.device('cuda'), CONFIG);assert digest == EXPECTED_SHA
    prep_settings = load_lq_preprocess_config(CONFIG);assert not prep_settings['enabled']
    runner = NativeY4UVTileRunner(network, torch.device('cuda'), prep_settings)
    provenance = {str(p): sha(p) for p in [Path(__file__), CONFIG, ROOT/'tools/infer_basic_text_line_full_image.py',
                  ROOT/'tools/basic_v2_eval_network.py', ROOT/'realesrgan/data/basic_text_line_full_frame.py', OUT/'code/manifest.json']}
    for folder in ['model', 'final', 'traces', 'receipts', 'previews']:(OUT/folder).mkdir(parents=True, exist_ok=True)
    rows = json.loads((OUT/'inventory.json').read_text())['images'][:limit]
    for row in rows[rank::shards]:
        started = time.perf_counter();stem = Path(row['name']).stem;ready = OUT/'prepared'/(stem+'.json')
        deadline = time.monotonic()+7200
        while not ready.exists():
            if time.monotonic() > deadline:raise TimeoutError(ready)
            time.sleep(2)
        prep = json.loads(ready.read_text());assert prep['source_sha256'] == row['sha256']
        source_path = Path(prep['source']);input_path = Path(prep['preprocessed']);ocr_path = Path(prep['ocr'])
        assert sha(source_path) == row['sha256'] and sha(input_path) == prep['preprocessed_sha256']
        assert sha(ocr_path) == prep['ocr_sha256']
        model_path = OUT/'model'/(stem+'.png');final_path = OUT/'final'/(stem+'.png')
        trace = infer_full_image(input_path, ocr_path, CHECKPOINT, model_path, device_name='cuda', network=network,
                    checkpoint_sha256=digest, network_config=CONFIG, tile_runner=runner, ocr_format='p17_annotations',
                    resize_mode='p17_height96_tile_pad', native_tile_height_threshold=128, large_tile_canvas_hw=(96, 480),
                    large_tile_overlap=64, large_tile_overlap_x=64, large_tile_overlap_y=24,
                    overlap=64, blend_radius=8, p17_vertical_context=.12, p17_horizontal_context=.03)
        save(OUT/'traces'/(stem+'.json'), trace)
        source = cv2.imread(str(source_path));image = cv2.imread(str(input_path));model = cv2.imread(str(model_path))
        assert source is not None and image is not None and model is not None and source.shape == image.shape == model.shape
        boxes = [dict(bbox_xyxy=r['ocr_box']) for r in prep['regions']]
        final, reports, frame_weight, frame_report = post.apply_postprocess(model, boxes, background_sigma=4., ambiguity_mode='conservative_edge')
        assert final.shape == source.shape and np.isfinite(final).all()
        assert cv2.imwrite(str(final_path), final, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        scale = min(1, 960/max(source.shape[:2]));size = (round(source.shape[1]*scale), round(source.shape[0]*scale))
        for label, pixels in [('source', source), ('preprocessed', image), ('model', model), ('final', final)]:
            assert cv2.imwrite(str(OUT/'previews'/(row['id']+'_'+label+'.jpg')), cv2.resize(pixels, size, interpolation=cv2.INTER_AREA))
        record = dict(id=row['id'], name=row['name'], source=str(source_path), source_sha256=row['sha256'],
                      preprocessed=str(input_path), preprocessed_sha256=prep['preprocessed_sha256'], width=source.shape[1], height=source.shape[0],
                      model=str(model_path), model_sha256=sha(model_path), output=str(final_path), output_sha256=sha(final_path),
                      checkpoint_sha256=digest, provenance=provenance, ocr_sha256=prep['ocr_sha256'],
                      text_boxes=[r['ocr_box'] for r in prep['regions']], postprocess=reports, board_protection=frame_report,
                      postprocess_changed_pixels=int(np.any(final != model, axis=2).sum()),
                      lines_processed=trace['lines_processed'], segments_total=trace['segments_total'],
                      seconds=time.perf_counter()-started)
        save(OUT/'receipts'/(stem+'.json'), record)
        print('RESTORED', row['id'], 'lines', record['lines_processed'], 'segments', record['segments_total'], round(time.perf_counter()-started, 2), flush=True)


def supervise():
    """@luna-comment
    Input:validated pilot, original280 inventory and available CPU/GPU slots.
    Output:stage progress, ten owned workers, complete hashed historical-run summary.
    Key Steps:six CPU OCR/preprocess workers plus four GPU workers; explicit failure
    stops only this run; require280 sources/stages/final hashes and clean exits.
    Key Params:CPU4threads each, GPUs0..3, two torch CPU threads each.
    State/Dependencies:isolated process group/logs; historical code/source unmodified.
    """
    processes = []
    for mode, python in [('prepare', '/home/guang.yang/deployTfLite/bin/python'), ('infer', '/home/guang.yang/realSR/bin/python')]:
        shards = 6 if mode == 'prepare' else 4
        for rank in range(shards):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4' if mode == 'prepare' else '2',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4' if mode == 'prepare' else '2', PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
            with (OUT/f'{mode}_{rank}.log').open('ab') as stream:
                child = subprocess.Popen([python, '-u', __file__, '--mode', mode, '--rank', str(rank), '--shards', str(shards)],
                                         cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((mode, rank, child))
    save(OUT/'launch.json', dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), jobs=[dict(mode=m, rank=r, pid=p.pid) for m, r, p in processes]))
    while True:
        states = [dict(mode=m, rank=r, pid=p.pid, exit_code=p.poll()) for m, r, p in processes]
        failed = any(s['exit_code'] not in (None, 0) for s in states)
        save(OUT/'status.json', dict(status='failed' if failed else 'running', expected=280,
                    prepared=len(list((OUT/'prepared').glob('*.json'))), completed=len(list((OUT/'receipts').glob('*.json'))),
                    jobs=states, updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        if failed:
            for _, _, child in processes:
                if child.poll() is None:child.terminate()
            raise RuntimeError(states)
        if all(s['exit_code'] == 0 for s in states):break
        time.sleep(15)
    inventory = json.loads((OUT/'inventory.json').read_text())['images']
    records = [json.loads((OUT/'receipts'/(Path(r['name']).stem+'.json')).read_text()) for r in inventory]
    assert len(records) == len({r['id'] for r in records}) == 280
    for row, item in zip(inventory, records):
        assert row['sha256'] == item['source_sha256'] and item['checkpoint_sha256'] == EXPECTED_SHA
        for path_key, digest_key in [('source', 'source_sha256'), ('preprocessed', 'preprocessed_sha256'), ('model', 'model_sha256'), ('output', 'output_sha256')]:
            assert sha(item[path_key]) == item[digest_key]
    save(OUT/'summary.json', dict(status='passed', images=280, checkpoint=str(CHECKPOINT), checkpoint_sha256=EXPECTED_SHA,
              pipeline='historical_tinyocr_v5borderrail_h96_normal_fullpost', total_lines=sum(r['lines_processed'] for r in records), results=records))
    save(OUT/'status.json', dict(status='passed', expected=280, prepared=280, completed=280, jobs=states, completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print('PASS280 historical pipeline', flush=True)


def main():
    """@luna-comment Input:explicit stage/shard/pilot CLI. Output:chosen pipeline stage.
    Key Steps:validate indices and dispatch. Key Params:280default, four formal shards.
    State/Dependencies:versioned constants and task-local manifests, no implicit downloads.
    """
    parser = argparse.ArgumentParser();parser.add_argument('--mode', choices=['prepare', 'infer', 'run'], default='run')
    parser.add_argument('--rank', type=int, default=0);parser.add_argument('--shards', type=int, default=1);parser.add_argument('--limit', type=int, default=280)
    args = parser.parse_args();assert 0 <= args.rank < args.shards
    if args.mode == 'run':supervise()
    elif args.mode == 'prepare':prepare(args.rank, args.shards, args.limit)
    else:infer(args.rank, args.shards, args.limit)


if __name__ == '__main__':main()
