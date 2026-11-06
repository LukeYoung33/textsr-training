"""Generate 24 paired PNG-derived-Y contrast controls from exact gallery finals."""
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
from tools.native_y_curve_four_images import TRIAL, process_pair


MANIFEST = ROOT/'output/camera280_six_block_comparison_20261026/manifest.json'
DESTINATION = ROOT/'output/camera280_six_block_comparison_20261026/postprocess_4_20261026'
CPP_REFERENCE = Path('E:/workspace/AISR/TextSR_Tiled_SVN/ImgProcLib/YUVCurveFilter.cpp')
IDS = ('051', '052', '099', '100')
MODELS = ('original_v3', 'original_v5', 'vosr_v3', 'vosr_v5', 'vosr2u_v3', 'vosr2u_v5')


def digest(data):
    """@luna-comment Input: immutable file bytes. Output: SHA256 hex. Key Steps: hash exact bytes before decode or after encode. Key Params: SHA256. State/Dependencies: no filesystem write."""
    return hashlib.sha256(data).hexdigest()


def file_url_path(url):
    """@luna-comment Input: exact manifest file URL. Output: Windows Y source Path. Key Steps: require file scheme, expected drive and PNG suffix; percent-decode only URL path. Key Params: Y source read-only. State/Dependencies: never rewrite source URL or select a different baseline."""
    parsed = urlparse(url)
    if parsed.scheme != 'file' or parsed.netloc or not parsed.path.lower().startswith('/y:/'):
        raise ValueError('Unexpected source URL '+url)
    path = Path(unquote(parsed.path.lstrip('/')))
    if path.suffix.lower() != '.png':
        raise ValueError('Expected PNG source '+url)
    return path


def encode_png(image):
    """@luna-comment Input: one native BGR8 result. Output: PNG bytes and host encode time. Key Steps: OpenCV PNG level3 encode. Key Params: no resize or display sharpening. State/Dependencies: encoding time excludes numerical core timing."""
    start = time.perf_counter()
    ok, data = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError('PNG encode failed')
    return data.tobytes(), time.perf_counter()-start


def generate_pair(identity, model, row, binding):
    """@luna-comment Input: exact four-row gallery entry/model and frozen source/code binding. Output: current/proposed PNGs and one pair receipt. Key Steps: hash/decode source once, shared background paired algorithm, native-size contract, exclusive writes, rehash source. Key Params: 051/052/099/100 and six keys only. State/Dependencies: writes only fresh repository output subtree."""
    url = row['images'][model]
    source_path = file_url_path(url)
    start = time.perf_counter()
    source_bytes = source_path.read_bytes()
    source_sha = digest(source_bytes)
    source = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    if source is None or source.dtype != np.uint8 or source.shape != (row['height'], row['width'], 3):
        raise AssertionError('Source decode/native-size mismatch '+model+' '+identity)
    decode_seconds = time.perf_counter()-start
    del source_bytes
    current, proposed, branch = process_pair(source, TRIAL)
    if current.shape != source.shape or proposed.shape != source.shape:
        raise AssertionError('Output native-size mismatch')
    current_bytes, current_encode_seconds = encode_png(current)
    proposed_bytes, proposed_encode_seconds = encode_png(proposed)
    name = model+'_'+identity
    current_path = DESTINATION/'current'/(name+'.png')
    proposed_path = DESTINATION/'proposed'/(name+'.png')
    receipt_path = DESTINATION/'receipts'/(name+'.json')
    for path in (current_path, proposed_path, receipt_path):
        if path.exists():
            raise FileExistsError('Pilot output already exists: '+str(path))
    if digest(source_path.read_bytes()) != source_sha:
        raise AssertionError('Original source changed during pair generation '+name)
    with current_path.open('xb') as stream:
        stream.write(current_bytes)
    with proposed_path.open('xb') as stream:
        stream.write(proposed_bytes)
    record = {'id': identity, 'model': model, 'domain': 'PNG-derived BT.601 full-range Y8 offline preview',
              'native_yuv_available': False, 'native_uv_byte_identity_proven': False,
              'neon_byte_identity_proven': False, 'mobile_latency_proven': False,
              'source_url': url, 'source_sha256': source_sha,
              'source_hw': [row['height'], row['width']],
              'source_unchanged_after_generation': True,
              'binding_sha256': binding,
              'trial': {'r0': TRIAL.r0, 'r1': TRIAL.r1, 'eta': TRIAL.eta, 'delta_d': TRIAL.delta_d},
              'current_png': str(current_path), 'current_sha256': digest(current_bytes),
              'proposed_png': str(proposed_path), 'proposed_sha256': digest(proposed_bytes),
              'host_timing_seconds': {'decode_input': decode_seconds,
                  'algorithm_background_and_derive_y': branch['background_seconds_host'],
                  'algorithm_paired_curve_and_rgb_preview': branch['paired_curve_and_preview_seconds_host'],
                  'current_png_encode': current_encode_seconds,
                  'proposed_png_encode': proposed_encode_seconds},
              'branch': branch}
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record


def main():
    """@luna-comment Input: frozen six-model gallery manifest and 24 exact PNG finals. Output: 48 new paired PNGs, 24 receipts and summary JSON. Key Steps: preflight identities/paths, hash code/C++ reference, serial one-image pairs, verify no source mutation and frozen code before summary. Key Params: one OpenCV worker and fixed exploratory trial. State/Dependencies: source gallery and old results read-only; no inference or native-device timing."""
    cv2.setNumThreads(1)
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = {row['id']: row for row in manifest['rows'] if row['id'] in IDS}
    if set(rows) != set(IDS) or len(rows) != len(IDS):
        raise AssertionError('Incorrect four-image identity set')
    for identity in IDS:
        if any(key not in rows[identity]['images'] for key in MODELS):
            raise AssertionError('Missing six-model source '+identity)
    all_paths = [file_url_path(rows[identity]['images'][model]) for identity in IDS for model in MODELS]
    if len(set(all_paths)) != 24 or not all(path.is_file() for path in all_paths):
        raise AssertionError('24 unique source PNGs required')
    targets = [DESTINATION/k/(m+'_'+i+suffix) for i in IDS for m in MODELS
               for k,suffix in (('current','.png'),('proposed','.png'),('receipts','.json'))]
    if any(path.exists() for path in targets) or (DESTINATION/'summary.json').exists():
        raise FileExistsError('Fresh output subtree required')
    binding_paths = {'cpp_reference': CPP_REFERENCE,
                     'python_core': ROOT/'tools/native_y_curve_four_images.py',
                     'python_runner': Path(__file__).resolve(),
                     'source_manifest': MANIFEST}
    binding = {key: digest(path.read_bytes()) for key,path in binding_paths.items()}
    for leaf in ('current','proposed','receipts'):
        (DESTINATION/leaf).mkdir(parents=True, exist_ok=True)
    records = []
    for identity in IDS:
        for model in MODELS:
            record = generate_pair(identity, model, rows[identity], binding)
            records.append(record)
            print(json.dumps({'done': model+'_'+identity,
                              'current_sha256': record['current_sha256'],
                              'proposed_sha256': record['proposed_sha256'],
                              'integer_pair_hits': record['branch']['changed_pixels_proposed_from_current']}), flush=True)
    if any(digest(path.read_bytes()) != binding[key] for key,path in binding_paths.items()):
        raise AssertionError('Code, C++ reference or source manifest changed during run')
    if len(records) != 24:
        raise AssertionError('Wrong pair count')
    summary = {'status': 'experimental_visual_pending', 'ids': IDS, 'models': MODELS,
               'domain': 'PNG-derived Y8; RGB-clipped preview; not native YUV420 byte proof',
               'source_manifest_sha256': digest(manifest_bytes), 'binding_sha256': binding,
               'pair_count': 24, 'new_png_count': 48, 'records': records}
    with (DESTINATION/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(str(DESTINATION/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
