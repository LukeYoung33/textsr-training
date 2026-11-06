"""Hash the 840 completed trial PNGs against their receipts on Remote58."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path('/AI_DNS/guang.yang/TextSR/data/camera_post_v2_20261023/full')
KEYS = ('light_v1_standard', 'light_v2_low', 'light_v2_standard')


def verify(item):
    """@luna-comment Input: one exact trial key and camera ID. Output: SHA256 verification record. Key Steps: stream PNG, compare with sibling receipt. Key Params: no decode or source/model execution. State/Dependencies: read-only task-owned image and receipt."""
    key, identity = item
    path = ROOT/key/(identity+'.png')
    receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    checksum = sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            checksum.update(block)
    actual = checksum.hexdigest()
    if receipt['id'] != identity or receipt['key'] != key or actual != receipt['final_sha256']:
        raise AssertionError('PNG/receipt mismatch '+key+' '+identity)
    return {'key': key, 'id': identity, 'sha256': actual, 'bytes': path.stat().st_size}


def main():
    """@luna-comment Input: 280x3 task-owned files. Output: approved-Y complete PNG hash audit JSON. Key Steps: eight independent streaming SHA workers then write only after every result passes. Key Params: 8 threads, immutable completed images. State/Dependencies: Remote58 existing Python only; no package install."""
    items = [(key, f'{identity:03d}') for identity in range(1, 281) for key in KEYS]
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(verify, items))
    destination = ROOT/'evaluation/remote_final_hash_audit.json'
    destination.write_text(json.dumps({'rows': 280, 'files': len(records), 'all_sha256_match_receipts': True,
                                       'records': records}, indent=2), encoding='utf-8')
    print(json.dumps({'rows': 280, 'files': len(records), 'all_sha256_match_receipts': True,
                      'total_bytes': sum(v['bytes'] for v in records)}))


if __name__ == '__main__':
    main()
