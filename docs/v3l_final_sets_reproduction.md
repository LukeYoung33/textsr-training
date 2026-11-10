# V3L final-set evaluation source inventory

## Scope
The historical Mojito32 and Titians75 run used `tools/evaluate_v3l_final_sets.py`.
The repository copy is byte-identical to the executed driver at
`/AI_DNS/guang.yang/TextSR/data/v3l_final500k_mojito32_titians75_20261109/evaluate_v3l_final_sets.py`.
The driver evaluates eight arms (four sizes, L1-R1 and L1-only), producing raw
and postprocessed outputs. The four-teacher export selects L1-R1 raw outputs only.

## Runtime and assets
Run on Linux with the existing PyTorch/CUDA training environment. `fcntl` is
required. The historical driver intentionally retains its frozen absolute paths:
- DATA: `/AI_DNS/guang.yang/TextSR/data`
- TRAIN: `v3l_s22_scratch_20261107` beneath DATA
- KIT: TRAIN/kit; this must contain the matching training architecture and tools.
- OUT: `v3l_final500k_mojito32_titians75_20261109` beneath DATA

Source synchronization alone does not include weights, source images, cached V3
inputs, selection JSON, or training completion receipts. Preserve those external
assets and their hashes. Do not blindly rerun the controller in an active output
folder: it queues all eight arms and audits all 1712 expected outputs.

```bash
python -B tools/evaluate_v3l_final_sets.py --help
# After provisioning the historical asset layout and matching KIT:
python -B -u tools/evaluate_v3l_final_sets.py run
# Existing prepared bindings, one selected arm, explicitly assigned GPU:
CUDA_VISIBLE_DEVICES=0 python -B -u tools/evaluate_v3l_final_sets.py worker --arm v3l16_L1_R1_ori
```

## V3 preprocessing dependency chain
`tools/prepare_v3pre_training.py` -> `neighbor.py` -> `revision2.py` -> `pre640.py`.
`revision2.py` also imports `tools/adaptive_scurve_robust_v2.py`, which imports
`evaluation/adaptive_prepost_frozen_v1_20261020/lowres_curve.py` and `postprocess.py`.
These source-only evaluation files are tracked; `.gitattributes` preserves their
frozen bytes on checkout. No generated pictures or caches are included.
The five frozen reference hashes in `prepare_v3pre_training.py` were verified.
The historical final-set inference reuses cached V3 inputs and does not rerun OCR
or preprocessing. Native inference uses patch256/stride224/crop8/smooth16.

## Validation
```bash
python -m pytest -o addopts= -q evaluation/document_box_neighbor_20261027/test_neighbor.py evaluation/document_box_pre640_revision2_20261027/test_revision2.py evaluation/document_box_pre640_astra_20261027/test_pre640.py
python -m compileall -q basicsr realesrgan tools
python tools/lenovo_stage_a_preprocess_screen.py --help
```
`-o addopts=` avoids the repository-wide `tests/` collection configured in setup.cfg.
See `v3l_evaluation_source_manifest.json` for exact SHA256 values. No algorithm
functions were changed during source recovery.
