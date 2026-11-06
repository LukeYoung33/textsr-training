"""Copy only verified identical-route V14 successes into a fresh batch."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    """@luna-comment Input:old/new batch directories. Output:verified reused receipts.
    Steps:exact code/job comparison, source/output hashes, lossless decode, copy.
    Params:no overwrite. Dependencies:local/shared files, no algorithm execution.
    """
    import cv2
    import numpy as np
    old, new = map(Path, sys.argv[1:])
    before = json.loads((old/'manifest.json').read_text())
    after = json.loads((new/'manifest.json').read_text())
    assert before['code_hashes'] == after['code_hashes']
    reused = []
    for i, job in enumerate(after['jobs']):
        prior = old/'receipts'/f'{i:03d}.json'
        if before['jobs'][i] != job or not prior.exists():
            continue
        receipt = json.loads(prior.read_text())
        if receipt['state'] != 'complete':
            continue
        image = old/job['output']
        assert hashlib.sha256(image.read_bytes()).hexdigest() == receipt['output_sha256']
        assert hashlib.sha256(Path(job['source']).read_bytes()).hexdigest() == job['sha256']
        decoded = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
        assert list(decoded.shape) == job['shape']
        dest = new/job['output']
        assert not dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dest)
        for sub in ('debug', 'previews'):
            relative = Path(sub)/job['group']/Path(job['source']).stem
            if sub == 'previews':
                relative = relative.with_suffix('.png')
            if (old/relative).is_dir():
                shutil.copytree(old/relative, new/relative)
            elif (old/relative).exists():
                (new/relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old/relative, new/relative)
        receipt['reused_from'] = str(prior)
        target = new/'receipts'/prior.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt, indent=2))
        reused.append(i)
    (new/'reuse.json').write_text(json.dumps(reused))
    print('REUSED', len(reused), reused, flush=True)


if __name__ == '__main__':
    main()
