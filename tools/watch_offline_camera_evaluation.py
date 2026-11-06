"""Read-only progress and final artifact validation for camera evaluation."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_offline_camera_matrix import ARMS, save_json, sha, verify_frozen


def main():
    """@luna-comment
    Input: approved camera evaluation run. Output: refreshing progress and validation receipt.
    Key Steps: read per-arm progress; await all four results and original-source proof;
    check every receipt/link and representative lossless PNG decode/hash, then publish.
    Key Params:30s polls; no training/evaluation process mutation.
    State/Dependencies: read-only outputs except own status/validation files.
    """
    import cv2
    output = ROOT.parent
    while True:
        pipeline_path = output / 'pipeline_status.json'
        pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else dict(phase='starting')
        progress = {}
        for arm in ARMS:
            path = output / arm / 'progress.json'
            progress[arm] = json.loads(path.read_text()) if path.exists() else dict(done=0, expected=280)
        source_proof = output / 'source_verification.json'
        ready = pipeline['phase'] == 'complete' and source_proof.exists()
        rows = ''.join(f'<tr><td>{arm}</td><td>{p["done"]} / {p["expected"]}</td></tr>' for arm,p in progress.items())
        page = '<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Camera evaluation progress</title><style>body{font:16px Segoe UI;margin:40px;color:#183247}td,th{padding:12px;border-bottom:1px solid #ddd}button{padding:10px 20px}</style><h1>Camera evaluation</h1><p>Phase: ' + pipeline['phase'] + '</p><table><tr><th>Arm</th><th>Images</th></tr>' + rows + '</table><p>Original source hashes: ' + ('verified' if source_proof.exists() else 'verification in progress') + '</p><button onclick="location.reload()">Refresh now</button><p>Automatic refresh every30 seconds.</p>'
        if ready:
            page += '<p><a href="index.html">Open completed comparison gallery</a></p>'
        (output / 'status.html').write_text(page, encoding='utf-8')
        save_json(output / 'live_progress.json', dict(pipeline=pipeline, arms=progress, original_source_verified=source_proof.exists()))
        if pipeline['phase'] == 'failed':
            return
        if ready:
            proof = json.loads(source_proof.read_text())
            assert proof['all_match'] and proof['images'] == 280
            frozen_hash = verify_frozen()
            count = 0
            checkpoints = {}
            for arm in ARMS:
                complete = json.loads((output / arm / 'COMPLETE.json').read_text())
                assert complete['complete'] and complete['images'] == 280
                checkpoints[arm] = complete['checkpoint_sha256']
                receipts = sorted((output / arm / 'receipts').glob('*.json'))
                assert len(receipts) == 280
                for path in receipts:
                    receipt = json.loads(path.read_text())
                    assert receipt['outside_unchanged'] and receipt['checkpoint_sha256'] == complete['checkpoint_sha256']
                    assert receipt['frozen_manifest_sha256'] == frozen_hash and receipt['postprocess_gamma'] == 1.2
                    assert (output / receipt['final']).is_file() and (output / receipt['model']).is_file()
                    if receipt['id'] in ('001','002','089','093','280'):
                        final = output / receipt['final']
                        assert sha(final) == receipt['final_sha256']
                        pixels = cv2.imread(str(final))
                        assert pixels is not None and list(pixels.shape[:2]) == receipt['source_hw']
                    count += 1
            assert (output / 'index.html').is_file()
            save_json(output / 'VALIDATED.json', dict(passed=True, images=280, outputs=count, checkpoint_sha256=checkpoints,
                       original_sources_match=True, frozen_manifest_sha256=frozen_hash, post_gamma=1.2,
                       representative_native_png_decodes=20, all_receipts_outside_unchanged=True))
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
