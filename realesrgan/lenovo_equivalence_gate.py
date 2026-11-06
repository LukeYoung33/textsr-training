"""Fail-closed content binding for formal Lenovo runs, on local or remote host."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(config, config_path):
    if 'lenovo_degradation' not in config or int(config.get('train', {}).get('total_iter', 0)) < 250000:
        return
    root = Path(__file__).resolve().parents[1]
    receipt = config.get('lenovo_equivalence_receipt')
    if not receipt:
        raise RuntimeError('Lenovo 250K refused: missing atomic equivalence receipt')
    receipt_path = root / receipt
    data = json.loads(receipt_path.read_text(encoding='utf-8'))
    if not all(data.get(key) is True for key in ('atomic_pass', 'c3_semantic_pass', 'composition_visual_pass', 'train_validation_schema_pass')):
        raise RuntimeError('Lenovo 250K refused: incomplete equivalence/visual acceptance')
    if data.get('config_sha256') != sha(config_path):
        raise RuntimeError('Lenovo 250K refused: configuration hash mismatch')
    files = data.get('files', {})
    required = {'realesrgan/lenovo_atomic.py', 'realesrgan/lenovo_atomic_torch.py', 'realesrgan/lenovo_c3.py', 'realesrgan/utils/lenovo_degradation.py',
                'realesrgan/models/lenovo_text_line_degradation_model.py',
                'datasets/lenovo_atomic_equivalence_v1/manifest.json'}
    if not required.issubset(files):
        raise RuntimeError('Lenovo 250K refused: missing code/golden manifest hashes')
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise RuntimeError(f'Lenovo 250K refused: deployment mismatch: {name}')
    manifest = json.loads((root/'datasets/lenovo_atomic_equivalence_v1/manifest.json').read_text(encoding='utf-8'))
    for record in manifest['records']:
        for key in ('gt', 'lq'):
            if sha(root/'datasets/lenovo_atomic_equivalence_v1'/record[key]) != record[key+'_sha256']:
                raise RuntimeError('Lenovo 250K refused: golden artifact mismatch')
