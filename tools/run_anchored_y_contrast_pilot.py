"""Run bounded PNG-derived-Y research arms on nine exact source finals."""
import argparse
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
from tools.anchored_y_contrast import ARMS, process_arm


OUTPUT = ROOT/'output/anchored_y_contrast_pilot_20261028'
CAMERA_MANIFEST = ROOT/'output/camera280_six_block_comparison_20261026/manifest.json'
PS_ROOT = ROOT/'debug/去灰且加对比'
CAMERA_IDS = ('006', '017', '051', '052', '099', '100')
PS_IDS = ('150', '151', '152')


def sha256(path):
    """@luna-comment Input: existing path. Output: exact file SHA256. Key Steps: stream bounded chunks. Key Params: SHA256. State/Dependencies: read-only source or output verification."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_url_path(url):
    """@luna-comment Input: manifest file URL. Output: exact Windows Y Path. Key Steps: validate Y file scheme and PNG suffix; decode percent escapes. Key Params: read-only Y source. State/Dependencies: never substitutes another model or geometry."""
    parsed = urlparse(url)
    if parsed.scheme != 'file' or parsed.netloc or not parsed.path.lower().startswith('/y:/'):
        raise ValueError('Unexpected camera source URL: '+url)
    path = Path(unquote(parsed.path.lstrip('/')))
    if path.suffix.lower() != '.png':
        raise ValueError('Expected PNG source: '+url)
    return path


def source_cases(manifest):
    """@luna-comment Input: frozen six-model manifest. Output: nine bound source/reference case dictionaries. Key Steps: PS V3M before sources and exact vosr_v5 camera rows. Key Params: 3PS+6camera. State/Dependencies: PS unsuffixed references are never fitted or processed."""
    rows = {row['id']: row for row in manifest['rows'] if row['id'] in CAMERA_IDS}
    if set(rows) != set(CAMERA_IDS):
        raise AssertionError('Camera identity set mismatch')
    cases = []
    for identity in PS_IDS:
        source = PS_ROOT/(identity+'_V3M.png')
        reference = PS_ROOT/(identity+'.png')
        lq = PS_ROOT/(identity+'_LQ.jpg')
        if not source.is_file() or not reference.is_file() or not lq.is_file():
            raise FileNotFoundError('Missing PS before/reference '+identity)
        cases.append({'key': 'ps'+identity, 'id': identity, 'name': 'PS '+identity,
                      'source_path': source, 'source_url': source.as_uri(),
                      'lq_url': lq.as_uri(), 'ps_reference_url': reference.as_uri(),
                      'expected_hw': (3120, 4160)})
    for identity in CAMERA_IDS:
        row = rows[identity]
        url = row['images']['vosr_v5']
        cases.append({'key': 'cam'+identity, 'id': identity, 'name': 'Camera '+identity,
                      'source_path': file_url_path(url), 'source_url': url,
                      'lq_url': row['lq'], 'ps_reference_url': None,
                      'expected_hw': (row['height'], row['width'])})
    if len({case['source_path'] for case in cases}) != 9:
        raise AssertionError('Nine unique source paths required')
    return cases


def encode_png(image):
    """@luna-comment Input: native BGR8 preview. Output: PNG bytes and host encoding seconds. Key Steps: OpenCV level3 PNG. Key Params: no resize or display filter. State/Dependencies: encoding excluded from curve compute timing."""
    began = time.perf_counter()
    ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return encoded.tobytes(), time.perf_counter()-began


def generate_case(case, arm_keys, bindings):
    """@luna-comment Input: one exact source, selected frozen arms and hash bindings. Output: exclusive PNGs/receipts and case summary. Key Steps: decode once, share current with first arm, rehash source, write only new task subtree. Key Params: native geometry and one cv thread. State/Dependencies: no source or reference write; current arm bound to frozen deployment."""
    path = case['source_path']
    began = time.perf_counter()
    source_bytes = path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    del source_bytes
    if source is None or source.dtype != np.uint8 or source.shape != (*case['expected_hw'], 3):
        raise AssertionError('Native source decode/geometry mismatch '+case['key'])
    decode_seconds = time.perf_counter()-began
    outputs = {}
    current_reference = None
    for arm_key in arm_keys:
        arm = ARMS[arm_key]
        current, candidate, telemetry = process_arm(source, arm)
        image = current if arm_key == 'current' else candidate
        if current_reference is None:
            current_reference = current
        elif not np.array_equal(current, current_reference):
            raise AssertionError('Current reference drift '+case['key'])
        if image.shape != source.shape:
            raise AssertionError('Output shape mismatch '+case['key']+' '+arm_key)
        png, encode_seconds = encode_png(image)
        output_path = OUTPUT/arm_key/(case['key']+'.png')
        receipt_path = OUTPUT/'receipts'/(case['key']+'_'+arm_key+'.json')
        if output_path.exists() or receipt_path.exists():
            raise FileExistsError('Existing output '+case['key']+' '+arm_key)
        if sha256(path) != source_hash:
            raise AssertionError('Source mutated during processing '+case['key'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('xb') as stream:
            stream.write(png)
        output_hash = hashlib.sha256(png).hexdigest()
        record = {'key': case['key'], 'id': case['id'], 'arm': arm_key,
                  'source_url': case['source_url'], 'source_sha256': source_hash,
                  'width': int(source.shape[1]), 'height': int(source.shape[0]),
                  'output_path': str(output_path), 'output_sha256': output_hash,
                  'bindings_sha256': bindings, 'arm_parameters': vars(arm),
                  'domain': 'PNG-derived BT.601 Y8, common signed BGR delta with RGB clipping',
                  'native_yuv_identity_proven': False, 'neon_identity_proven': False,
                  'device_latency_proven': False, 'source_unchanged_after_generation': True,
                  'decode_seconds_host': decode_seconds, 'encode_seconds_host': encode_seconds,
                  'telemetry': telemetry}
        with receipt_path.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        outputs[arm_key] = {'path': str(output_path), 'sha256': output_hash,
                            'receipt_path': str(receipt_path), 'changed_pixels_vs_current':
                            telemetry['changed_pixels_candidate_vs_current'] if arm_key != 'current' else 0}
        print(json.dumps({'done': case['key'], 'arm': arm_key,
                          'changed_pixels_vs_current': outputs[arm_key]['changed_pixels_vs_current']}), flush=True)
    return {'key': case['key'], 'id': case['id'], 'name': case['name'],
            'width': int(source.shape[1]), 'height': int(source.shape[0]),
            'source_url': case['source_url'], 'lq_url': case['lq_url'],
            'ps_reference_url': case['ps_reference_url'], 'source_sha256': source_hash,
            'outputs': outputs}


def main():
    """@luna-comment Input: arm CLI and nine preflighted sources. Output: staged research PNG/receipts/summary. Key Steps: source/code hash binding, serial one-image processing, exclusive writes, summary only actually generated arms. Key Params: default current+weak; stronger arms require parent gate. State/Dependencies: experimental host preview, no gallery mutation or quality claim."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('weak','full','strong'), default='weak')
    args = parser.parse_args()
    arms = {'weak':['current','weak'], 'full':['mean_g3','upper_g3'],
            'strong':['upper_g4']}[args.stage]
    cv2.setNumThreads(1)
    manifest_bytes = CAMERA_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    cases = source_cases(manifest)
    bindings = {'source_manifest': hashlib.sha256(manifest_bytes).hexdigest(),
                'core': sha256(ROOT/'tools/anchored_y_contrast.py'),
                'frozen_current': sha256(ROOT/'tools/native_y_curve_four_images.py'),
                'runner': sha256(Path(__file__).resolve())}
    summary_path = OUTPUT/(args.stage+'_summary.json')
    if summary_path.exists():
        raise FileExistsError('Existing stage summary '+str(summary_path))
    for case in cases:
        for arm in arms:
            if (OUTPUT/arm/(case['key']+'.png')).exists() or (OUTPUT/'receipts'/(case['key']+'_'+arm+'.json')).exists():
                raise FileExistsError('Existing output '+case['key']+' '+arm)
    records = [generate_case(case, arms, bindings) for case in cases]
    if (sha256(CAMERA_MANIFEST) != bindings['source_manifest'] or
            sha256(ROOT/'tools/anchored_y_contrast.py') != bindings['core'] or
            sha256(ROOT/'tools/native_y_curve_four_images.py') != bindings['frozen_current'] or
            sha256(Path(__file__).resolve()) != bindings['runner']):
        raise AssertionError('Source manifest or code changed during run')
    summary = {'status': 'research_preview_visual_pending', 'stage': args.stage, 'arms': arms,
               'skipped_arms': [key for key in ARMS if key not in arms],
               'bindings_sha256': bindings, 'domain': 'PNG-derived Y8 host preview; no native UV/NEON or 140ms proof',
               'case_count': len(records), 'new_png_count': len(records)*len(arms), 'cases': records}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with summary_path.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(str(summary_path), flush=True)


if __name__ == '__main__':
    main()
