# SG/WH data-free training kit

This is a source snapshot plus proposed training templates, not evidence of GPU training readiness.
Read docs/training/sg_wh_training_plan_20261106_CN.md for the Chinese plan.

## Contents and exclusions
- basicsr/, realesrgan/: Python libraries and custom-op sources; tools/: source utilities.
- configs/:25 supervised templates (SG5pretrain+10finetune;WH10scratch) and8 explicitly component-only KD specifications.
- tests/, options/, docs/, evaluation/: focused CPU tests, architecture fragments, plans and prior CPU evidence.
- STAGES.json, MANIFEST.json, bindings.example.json, dependency and environment references.
- No datasets, image/array shards, real sample manifests, VCS objects, cached files, installed environments, learned checkpoints or binaries.
- Random initial weights are generated once using the init command below, then that exact directory is shared between sites.
- Optional OCR/perceptual weights are not included; the supplied templates disable those objectives.

## Environment
Use a separate environment. Install a matching CUDA-capable torch/torchvision pair for the target GPU first, then the remaining requirements_runtime.txt dependencies. No installation is performed by this kit. environment_reference.json records local versions (CPU torch); it is not a GPU lockfile. The original requirements.txt is reference only: its pip basicsr requirement must not overwrite the bundled modified basicsr source. Run commands from this kit root; no editable package installation is required.

## Preparation commands
```sh
python tools/sg_wh_training_runtime.py check
python tools/sg_wh_training_runtime.py init --directory initializations --seed 20260828
python tools/profile_v3l_teachers.py
python -m pytest -q tests/test_v3l_feature_distillation.py tests/test_v3m_textboost.py --basetemp=.artifacts/pytest-kit
```
The init output is random initialization, not pretraining. The command refuses to overwrite existing weights. Generate once and copy those same files to both sites.

Copy bindings.example.json to a new JSON file and supply explicit absolute paths. PRETRAIN_TRAIN_MANIFEST,PRETRAIN_VAL_MANIFEST,NEW_TRAIN_MANIFEST,NEW_VAL_MANIFEST are explicit offline JSON manifests. Each bank root is its manifest parent; training and validation banks may be separate. INIT_ROOT holds V3M.pth,V3L16.pth,V3L32.pth,V3L64.pth,V3L128.pth. SG_PRETRAIN_ROOT holds selected SG pretraining exports with those names. WH_TEACHER_ROOT holds selected teachers named V3L16_L1.pth etc. RUN_ROOT is the experiment output parent. No example is bound to old machine paths.

```sh
python tools/sg_wh_training_runtime.py bind --template configs/WH_SCRATCH/V3M_L1.template.yml --bindings bindings.WH.json --output bound/V3M_L1.yml
python tools/sg_wh_training_runtime.py check --config bound/V3M_L1.yml
# Explicit training command: use only after data and CUDA acceptance.
python tools/sg_wh_training_runtime.py train -opt bound/V3M_L1.yml
```
The portable launcher uses existing Block256RegionModel/native losses and registers V3L explicitly. Templates use plan defaults; they still require real-data/CUDA validation and recipe review. L1-R1_ori is original Missing1/False0.25, not width-normalized R1. The original historical tools are included for dependencies/reference; do not launch their old machine-specific controllers. Use only the portable launcher for these templates.

## Data contract (data not supplied)
The supervised templates deliberately target the existing OfflineAugmentedRegionDataset reader. Each selected manifest parent must contain COMPLETE.json and the index/shard/metadata files referenced by that manifest, in the format produced by tools/build_offline_augmented_dataset.py. Shards are uint8[N,256,256,8]:BGR LQ,BGR GT,text mask,valid mask (the reader converts BGR to RGB tensors). Completion/build identities and canvas must match. A folder of loose RGB pairs is not interchangeable with this bank. No dataset conversion or regeneration is performed by this kit. If the separately organized data uses another format, bind the correct existing reader or integrate it explicitly before launch.

## KD readiness
realesrgan/models/v3_feature_distillation.py contains the tested8-tap objective and resumable single-device step engine. configs/WH_KD/*.component.yml are not full executable training configs. Base-loss closure binding (especially R1/masks), real data/sampler/resume and required CUDA/AMP/DDP integration remain to be completed before formal KD. The launcher explicitly refuses KD component configs. Feature weight1 is untuned; follow the20K pilot in the plan. No trained teacher checkpoint currently belongs to this delivery.

## Integrity and acceptance
MANIFEST.json lists SHA256 for included files. VALIDATION.json and an external archive SHA256 receipt describe checks of the delivered copy. Prior CPU evidence under evaluation/ predates packaging and must not be mistaken for target-site CUDA validation. No training or network operations run automatically on extraction.
