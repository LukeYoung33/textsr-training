"""Read-only inventory of VOSR target geometry, pairs and capture splits."""
import json
from collections import Counter
from pathlib import Path
import struct


def image_hw(path):
    """@luna-comment
    Input: PNG file path. Output: height/width from the authoritative IHDR.
    Key Steps: check PNG signature and unpack dimensions. Key Params:24 header bytes.
    State/Dependencies: read-only filesystem and struct; no pixel changes.
    """
    with path.open('rb') as stream:
        header = stream.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(str(path))
    width, height = struct.unpack('>II', header[16:24])
    return height, width


def main():
    """@luna-comment
    Input: fixed user-approved source root. Output: bounded JSON inventory.
    Key Steps: compare published row identities/files/dimensions and split membership.
    Key Params: all rows, five aligned assets. State/Dependencies: immutable source.
    """
    root = Path('/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261014/full_images')
    result, identities = {}, {}
    for split in ('train', 'val'):
        rows = [json.loads(line) for line in (root / 'manifests' / (split + '.jsonl')).read_text().splitlines() if line.strip()]
        files = {p.name for p in (root / split / 'GT_VOSR').iterdir() if p.is_file()}
        errors, examples = [], []
        identities[split] = {r['source_identity'] for r in rows}
        for row in rows:
            target = root / split / 'GT_VOSR' / Path(row['files']['gt']).name
            if not target.exists():
                errors.append([row['sample_id'], 'missing_target'])
                continue
            shapes = {k: image_hw(root / row['files'][k]) for k in ('lq', 'gt', 'text_mask', 'valid_mask')}
            shapes['vosr'] = image_hw(target)
            if len(set(shapes.values())) != 1:
                errors.append([row['sample_id'], shapes])
            if len(examples) < 4:
                examples.append(dict(id=row['sample_id'], shapes=shapes))
        result[split] = dict(rows=len(rows), vosr_files=len(files), families=dict(Counter(r['dataset'] for r in rows)), errors=errors[:20], error_count=len(errors), extras=sorted(files - {Path(r['files']['gt']).name for r in rows})[:20], examples=examples)
    result['capture_overlap'] = len(identities['train'] & identities['val'])
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
