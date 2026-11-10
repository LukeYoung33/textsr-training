"""Standalone integrity and numerical checks for this frozen package; no model needed."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
for relative, expected in manifest['files'].items():
    path = (HERE / relative).resolve()
    assert path.is_relative_to(HERE), relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative

pre = importlib.import_module(HERE.name + '.preprocess')
post = importlib.import_module(HERE.name + '.postprocess')
current = dict(paper=np.array([128.], np.float32), normalized_anchors=[20, 110, 200],
               centers=np.array([1.5], np.float32))
reference = dict(anchors=np.array([[10, 100, 190]], np.float32), centers=current['centers'])
linear = post.build_transfer_tables(current, reference, 1)
assert linear['tables'][0, [20, 110, 200]].tolist() == [10*256, 100*256, 190*256]
assert linear['tables'][0, 65] == 55*256
mild = post.build_transfer_tables(current, reference)
assert abs(int(mild['tables'][0, 65])/256 - 49.174775) < .01
mirror = {**current, 'normalized_anchors': [55, 145, 235]}
mirror_ref = {**reference, 'anchors': np.array([[65, 155, 245]], np.float32)}
mirrored = post.build_transfer_tables(mirror, mirror_ref)['tables'][0].astype(int)
assert np.max(abs(mild['tables'][0].astype(int) + mirrored[::-1] - 255*256)) <= 1
assert np.all(np.diff(mild['tables'].astype(int), axis=1) >= 0)
identity = {**current, 'normalized_anchors': [0, 127.5, 255]}
identity_ref = {**reference, 'anchors': np.array([[0, 127.5, 255]], np.float32)}
assert np.array_equal(post.build_transfer_tables(identity, identity_ref, 1)['tables'][0],
                      np.arange(256)*256)

source = np.full((96, 256, 3), 200, np.uint8)
source[24:72, ::16] = 40
source[24:72, 1::16] = 40
original = source.copy()
context = pre.prepare_context(source, [[16, 12, 240, 84]], reference_id='synthetic')
prepared = pre.apply_preprocess(source, context)
assert np.array_equal(source, original)
assert all(line['packed']['valid'].all() for line in context['lines'])
compact = json.loads(json.dumps(pre.make_post_context(context)))
shifted = np.clip(source.astype(int)+10, 0, 255).astype(np.uint8)
fixed, reports = post.postprocess_bgr(shifted, compact, gamma=1)
mask = np.zeros(source.shape[:2], bool)
mask[12:84, 16:240] = True
assert np.array_equal(fixed[~mask], shifted[~mask])
assert abs(float(np.median(fixed[:, :, 0][(source[:, :, 0] == 40) & mask]))-40) <= 1
assert abs(float(np.median(fixed[:, :, 0][(source[:, :, 0] == 200) & mask]))-200) <= 1
assert all(row['all_columns'] and row['target_reused'] and row['current_reestimated'] for row in reports)
empty = pre.prepare_context(source, [], reference_id='empty')
assert np.array_equal(pre.apply_preprocess(source, empty), source)
no_post, no_reports = post.postprocess_bgr(source, pre.make_post_context(empty))
assert np.array_equal(no_post, source) and not no_reports
print(json.dumps(dict(passed=True, version=manifest['version'], files_verified=len(manifest['files']),
                      anchor_mapping=True, monotone=True, polarity_symmetry=True,
                      known_shift_repair=True, context_json_roundtrip=True,
                      all_columns=True, outside_boxes_unchanged=True, empty_boxes=True)))
