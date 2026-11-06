"""Verify every camera viewer V5 source URL, receipt and document/OCR binding before batch."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_camera_light_post_v2 import BASE, FIX, MANIFEST, OUT, digest, mapped_path


def check():
    """@luna-comment Input: original 280 viewer manifest. Output: Y preflight audit with exact source and geometry hashes. Key Steps: resolve each row URL, sibling receipt, document record and OCR hash; require 017 accepted refresh. Key Params: 280 rows, V5-block key. State/Dependencies: read-only old Y, writes only approved new Y audit."""
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if len(manifest['rows']) != 280:
        raise AssertionError('Expected 280 originals')
    records = []
    for row in manifest['rows']:
        identity = row['id']
        source = mapped_path(row['images']['vosr_v5_block'])
        receipt_file = source.parent.parent/'receipts'/(identity+'.json')
        receipt = json.loads(receipt_file.read_text(encoding='utf-8'))
        if digest(source) != receipt['final_sha256']:
            raise AssertionError('Baseline hash '+identity)
        if identity == '017':
            if 'camera017_comparison_refresh_20261022' not in str(source):
                raise AssertionError('017 must use accepted refresh')
            record_file = FIX/'geometry_cache/documents/017.jpg/result.json'
            doc_root = FIX/'geometry_cache'
            fix = json.loads((FIX/'receipts/017.json').read_text(encoding='utf-8'))
            if receipt['document_record_sha256'] != fix['corrected_document_record_sha256']:
                raise AssertionError('017 geometry binding')
        else:
            if BASE not in source.parents:
                raise AssertionError('Unexpected source family '+identity)
            record_file = mapped_path(receipt['document_record'])
            doc_root = record_file.parents[2]
        if digest(record_file) != receipt['document_record_sha256']:
            raise AssertionError('Document record hash '+identity)
        stages = {v['instance_id']: v for v in receipt['stages']}
        for iid, stage in stages.items():
            ocr_file = doc_root/'ocr'/identity/(f'{iid:03d}.json')
            if digest(ocr_file) != stage['ocr_sha256']:
                raise AssertionError('OCR hash '+identity)
            ocr = json.loads(ocr_file.read_text(encoding='utf-8'))
            if ocr['source_sha256'] != receipt['source_sha256']:
                raise AssertionError('OCR source '+identity)
        records.append({'id': identity, 'baseline': str(source), 'baseline_sha256': receipt['final_sha256'], 'receipt_sha256': digest(receipt_file), 'document_record_sha256': receipt['document_record_sha256'], 'instances': len(stages)})
    target = OUT/'evaluation/source_preflight.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'rows': len(records), 'exceptions': ['017'], 'records': records}, indent=2), encoding='utf-8')
    return target


if __name__ == '__main__':
    print(check())
