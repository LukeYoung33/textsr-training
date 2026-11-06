"""Apply the frozen deployed-current PNG-derived Y curve to two HX52 arms."""
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_y_curve_four_images import CurveTrial, process_pair


GALLERY = ROOT/'output/hx52_finetuned_v3_v3m_0921_20261027'
MANIFEST = GALLERY/'manifest.json'
DESTINATION = GALLERY/'post_current_y_20261027'
MODELS = ('full_v3', 'text15_v3m')
CURRENT_ONLY = CurveTrial(16.0, 48.0, 0.0, 0.0)


def sha256_file(path):
    """@luna-comment Input: exact existing file path. Output: SHA256 of its bytes. Key Steps: streamed read for pre/post source and frozen-code binding. Key Params: SHA256. State/Dependencies: read-only."""
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def source_path(url):
    """@luna-comment Input: manifest source URL. Output: exact Y PNG path. Key Steps: require file scheme, Y drive and PNG suffix. Key Params: no alternate baseline lookup. State/Dependencies: approved related result trees are read-only."""
    parsed = urlparse(url)
    if parsed.scheme != 'file' or parsed.netloc or not parsed.path.lower().startswith('/y:/'):
        raise ValueError('Unexpected HX source URL '+url)
    path = Path(unquote(parsed.path.lstrip('/')))
    if path.suffix.lower() != '.png':
        raise ValueError('Expected final PNG '+url)
    return path


def load_manifest():
    """@luna-comment Input: frozen local HX52 gallery manifest. Output: exact 52-row list and SHA. Key Steps: verify ID uniqueness, required model URLs and native dimensions. Key Params: full_v3/text15_v3m only. State/Dependencies: no manifest mutation."""
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    rows = manifest['rows']
    if len(rows) != 52 or len({row['id'] for row in rows}) != 52:
        raise AssertionError('Expected exactly 52 unique HX rows')
    for row in rows:
        if any(key not in row['images'] for key in MODELS) or min(row['height'], row['width']) < 1:
            raise AssertionError('Incomplete HX source row '+row['id'])
    return rows, hashlib.sha256(raw).hexdigest()


def valid_existing_receipt(receipt_path, source_sha, bindings, output_path, row):
    """@luna-comment Input: optional existing own-task receipt and expected bindings. Output: verified record or None. Key Steps: reject partial/unbound output; rehash image before resume. Key Params: exact source/core/manifest SHA and dimensions. State/Dependencies: never overwrite an existing result."""
    if not receipt_path.exists() and not output_path.exists():
        return None
    if not receipt_path.is_file() or not output_path.is_file():
        raise AssertionError('Partial existing HX output '+str(output_path))
    record = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (record['source_sha256'] != source_sha or record['bindings_sha256'] != bindings
            or record['native_hw'] != [row['height'], row['width']]
            or record['output_sha256'] != sha256_file(output_path)):
        raise AssertionError('Existing HX receipt mismatch '+str(receipt_path))
    return record


def process_one(row, model, bindings):
    """@luna-comment Input: one exact source row/model and frozen SHA set. Output: one new native PNG and bound receipt. Key Steps: source hash/decode once, zero-trial frozen-current branch, equality/cap checks, exclusive encode/write, source rehash. Key Params: gain.75/deadband2/cap24. State/Dependencies: single-process, PNG-derived Y offline display only."""
    identity = row['id']
    url = row['images'][model]
    original = source_path(url)
    source_sha = sha256_file(original)
    output_path = DESTINATION/model/(identity+'.png')
    receipt_path = DESTINATION/'receipts'/(model+'_'+identity+'.json')
    old = valid_existing_receipt(receipt_path, source_sha, bindings, output_path, row)
    if old is not None:
        return old, True
    decode_started = time.perf_counter()
    source_bytes = original.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source_sha:
        raise AssertionError('HX source changed before decode '+model+' '+identity)
    source = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    if source is None or source.dtype != np.uint8 or source.shape != (row['height'], row['width'], 3):
        raise AssertionError('HX native decode mismatch '+model+' '+identity)
    decode_seconds = time.perf_counter()-decode_started
    del source_bytes
    current, zero_candidate, branch = process_pair(source, CURRENT_ONLY)
    if not np.array_equal(current, zero_candidate):
        raise AssertionError('Frozen current zero-switch mismatch '+model+' '+identity)
    del zero_candidate
    channel_change = np.abs(current.astype(np.int16)-source.astype(np.int16))
    max_change = int(channel_change.max())
    if max_change > 24 or current.shape != source.shape:
        raise AssertionError('HX current output cap/native-size mismatch '+model+' '+identity)
    encode_started = time.perf_counter()
    ok, encoded = cv2.imencode('.png', current, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError('HX PNG encode failed '+model+' '+identity)
    output_bytes = encoded.tobytes()
    encode_seconds = time.perf_counter()-encode_started
    if sha256_file(original) != source_sha:
        raise AssertionError('HX source changed during postprocess '+model+' '+identity)
    with output_path.open('xb') as stream:
        stream.write(output_bytes)
    record = {'id': identity, 'model': model, 'source_url': url,
              'source_sha256': source_sha, 'source_unchanged_after_generation': True,
              'output_path': str(output_path), 'output_sha256': hashlib.sha256(output_bytes).hexdigest(),
              'native_hw': [row['height'], row['width']], 'bindings_sha256': bindings,
              'domain': 'PNG-derived full-range BT.601 Y8 offline approximation',
              'native_yuv_or_neon_byte_identity_proven': False, 'mobile_latency_proven': False,
              'parameters': {'cell': 16, 'grid_smoothing': 'two separable replicated 1-2-1 rounds',
                             'sampling': 'half-pixel bilinear', 'polarity': 'smoothstep((B-80)/96)',
                             'deadband': 2, 'gain': .75, 'softcap': 24,
                             'candidate_eta': 0, 'candidate_delta_d': 0},
              'max_abs_bgr_channel_change': max_change,
              'zero_candidate_equals_current': True,
              'changed_pixels': branch['changed_pixels_current_from_source'],
              'rgb_clipped_channels': branch['rgb_clipped_channels_current'],
              'host_timing_seconds': {'decode': decode_seconds,
                  'derive_y_and_background': branch['background_seconds_host'],
                  'current_curve_and_rgb_preview': branch['paired_curve_and_preview_seconds_host'],
                  'png_encode': encode_seconds}}
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record, False


def main():
    """@luna-comment Input: 52-row gallery and exact two model-final sets. Output: 104 bounded native PNGs/receipts plus summary. Key Steps: freeze hashes, serial source-bound processing, verify core/manifest unchanged. Key Params: one OpenCV thread to protect low free RAM. State/Dependencies: old gallery, source finals and frozen core are read-only."""
    cv2.setNumThreads(1)
    rows, manifest_sha = load_manifest()
    bindings_paths = {'manifest': MANIFEST, 'core': ROOT/'tools/native_y_curve_four_images.py',
                      'runner': Path(__file__).resolve()}
    bindings = {key: sha256_file(path) for key,path in bindings_paths.items()}
    if bindings['manifest'] != manifest_sha:
        raise AssertionError('Manifest hash race')
    originals = [source_path(row['images'][model]) for row in rows for model in MODELS]
    if len(originals) != 104 or len(set(originals)) != 104 or not all(path.is_file() for path in originals):
        raise AssertionError('Expected 104 unique existing HX finals')
    for leaf in (*MODELS, 'receipts'):
        (DESTINATION/leaf).mkdir(parents=True, exist_ok=True)
    records = []
    resumed = 0
    for row in rows:
        for model in MODELS:
            record, skipped = process_one(row, model, bindings)
            records.append(record)
            resumed += int(skipped)
            print(json.dumps({'done': model+'_'+row['id'], 'resumed': skipped,
                              'changed_pixels': record['changed_pixels'],
                              'max_abs_channel_change': record['max_abs_bgr_channel_change']}), flush=True)
    if any(sha256_file(path) != bindings[key] for key,path in bindings_paths.items()):
        raise AssertionError('Frozen HX manifest/core/runner changed during batch')
    if len(records) != 104:
        raise AssertionError('Wrong HX output count')
    summary = {'status': 'experimental_visual_pending',
               'ids': [row['id'] for row in rows], 'models': MODELS,
               'record_count': len(records), 'resumed_count': resumed,
               'source_manifest_sha256': manifest_sha, 'bindings_sha256': bindings,
               'domain': 'PNG-derived Y8 offline approximation; not native YUV or NEON byte proof',
               'records': records}
    summary_path = DESTINATION/'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(summary_path), flush=True)


if __name__ == '__main__':
    main()
