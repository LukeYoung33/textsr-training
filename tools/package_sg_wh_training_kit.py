"""Assemble the authorized SG/WH source kit without data or learned checkpoints."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """@luna-comment
    Input: local file. Output: SHA256 hex digest. Key Steps: stream bounded chunks.
    Key Params: 1MiB chunks. State/Dependencies: read-only file bytes.
    """
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def supervised(model, stage, loss):
    """@luna-comment
    Input: architecture name, SG/WH stage and loss label. Output: portable unbound BasicSR template.
    Key Steps: use existing block/native loss path; disable unrelated auxiliary objectives.
    Key Params: provisional500K, native L1 plus optional original Missing1/False0.25.
    State/Dependencies: approved plan defaults; actual data and initial weights remain placeholders.
    """
    import yaml
    network = dict(type='BasicTextLineR4StudentY4UVV3M', canvas_hw=[256, 256])
    if model != 'V3M':
        network.update(type='BasicTextLineR4TeacherY4UVV3L', compute_ratio=int(model[3:]))
    data_prefix = 'PRETRAIN' if stage == 'SG_PRE' else 'NEW'
    initial_root = '${SG_PRETRAIN_ROOT}' if stage == 'SG_FT' else '${INIT_ROOT}'
    lr = 1e-5 if stage == 'SG_FT' else 1e-4
    data = {}
    for split in ('train', 'val'):
        data[split] = dict(name=f'{stage}_{split}', type='OfflineAugmentedRegionDataset',
                           manifest='${' + data_prefix + '_' + split.upper() + '_MANIFEST}',
                           canvas_hw=[256, 256], cache_shards=2,
                           batch_size_per_gpu=4 if split == 'train' else 1,
                           num_worker_per_gpu=2 if split == 'train' else 0)
    config = dict(name=f'{stage}_{model}_{loss}_500k', model_type='Block256RegionModel',
                  scale=1, num_gpu=1, manual_seed=20260828, paired_probability=1,
                  network_g=network, datasets=data,
                  path=dict(pretrain_network_g=f'{initial_root}/{model}.pth', strict_load_g=True,
                            param_key_g='params', resume_state=None, experiments_root='${RUN_ROOT}'),
                  train=dict(ema_decay=0, optim_g=dict(type='Adam', lr=lr, betas=[.9, .99]),
                             scheduler=dict(type='CosineAnnealingRestartLR', periods=[500000],
                                            restart_weights=[1], eta_min=lr * .01),
                             total_iter=500000, warmup_iter=5000, loss_mode='native',
                             train_reference_loss=False, validation_reference_loss=False,
                             y4uv_native_opt=dict(type='Y4UVNativeMaskedL1LossV3', loss_weight=1,
                                                  channel_weight=[1, 1, 1, 1, .25, .25],
                                                  fg_weight=1, bg_weight=1),
                             aux_start_iter=2000, aux_ramp_iters=5000,
                             cp_weight=0, grad_weight=0, bg_lf_weight=0, sdf_boundary_weight=0,
                             uv_strong_weight=0, recall_aware_weight_missing=1 if loss == 'L1_R1_ori' else 0,
                             recall_aware_weight_false=.25 if loss == 'L1_R1_ori' else 0,
                             recall_aware_opt=dict(ring_radius=3, margin=.03, temperature=.01,
                                                   min_polarity_contrast=.03) if loss == 'L1_R1_ori' else None),
                  val=dict(val_freq=50000, save_img=False, metrics={}),
                  logger=dict(print_freq=100, save_checkpoint_freq=25000, use_tb_logger=False),
                  dist_params=dict(backend='nccl', port=29517))
    return '# Proposed plan template; bind paths and validate real-data/CUDA behavior before launch.\n' + yaml.safe_dump(config, sort_keys=False)


def assemble(destination):
    """@luna-comment
    Input: new destination inside repository delivery. Output: source kit,33 templates and source hashes.
    Key Steps: copy allowlisted code extensions, selected docs/tests, generate configs/readiness metadata.
    Key Params: no symlinks, datasets, arrays, learned weights, binaries, caches or VCS payload.
    State/Dependencies: creates only new delivery artifacts; source files remain unchanged.
    """
    import yaml
    destination = destination.resolve()
    destination.relative_to(ROOT / 'delivery')
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    extensions = {'.py', '.cpp', '.cc', '.c', '.h', '.hpp', '.cu', '.cuh'}
    for folder in ('basicsr', 'realesrgan', 'tools'):
        for path in (ROOT / folder).rglob('*'):
            if path.suffix.lower() in extensions and path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                if any(part in ('third_party', 'datasets', 'weights', '.git', '.svn', 'share') for part in path.relative_to(ROOT).parts):
                    continue
                path.resolve().relative_to(ROOT)
                files.append(path)
    for name in ('LICENSE', 'requirements.txt', 'VERSION'):
        if (ROOT / name).is_file(): files.append(ROOT / name)
    for name in ('test_v3l_feature_distillation.py', 'test_v3m_textboost.py'):
        files.append(ROOT / 'tests' / name)
    for name in ('sg_wh_training_plan_20261106.md', 'sg_wh_training_plan_20261106_CN.md',
                 'v3l_feature_distillation_20261106.md', 'v3l_comment_review_20261106.md',
                 'sg_wh_training_kit_README_CN.md'):
        files.append(ROOT / 'docs/training' / name)
    for name in ('profiles.json', 'acceptance.json', 'tests.xml', 'tests.log'):
        files.append(ROOT / 'evaluation/v3l_feature_distillation_20261106' / name)
    files += list((ROOT / 'options/v3l_feature_distillation').glob('*.yml'))
    for source in sorted(set(files)):
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    stages = []
    for model in ('V3M', 'V3L16', 'V3L32', 'V3L64', 'V3L128'):
        for stage, losses in [('SG_PRE', ['L1']), ('SG_FT', ['L1', 'L1_R1_ori']), ('WH_SCRATCH', ['L1', 'L1_R1_ori'])]:
            for loss in losses:
                relative = Path('configs') / stage / f'{model}_{loss}.template.yml'
                path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(supervised(model, stage, loss), encoding='utf-8')
                stages.append(dict(id=f'{stage}_{model}_{loss}', kind='supervised_template', config=relative.as_posix(), ready='needs_path_binding_and_real_cuda_validation'))
    for ratio in (16, 32, 64, 128):
        for loss in ('L1', 'L1_R1_ori'):
            relative = Path('configs/WH_KD') / f'V3M_from_V3L{ratio}_{loss}.component.yml'
            path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(kind='feature_distillation_component', student='V3M', teacher=f'V3L{ratio}',
                           teacher_checkpoint=f'${{WH_TEACHER_ROOT}}/V3L{ratio}_{loss}.pth',
                           student_initialization='${INIT_ROOT}/V3M.pth', base_loss_recipe=loss,
                           train_manifest='${NEW_TRAIN_MANIFEST}', val_manifest='${NEW_VAL_MANIFEST}', total_iter=500000,
                           feature_weight=1, warmup_iters=5000, pilot_weights=[.01, .1, 1.],
                           stage_weights=[1]*8, integration_status='full_data_pipeline_not_integrated')
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
            stages.append(dict(id=f'WH_KD_V3L{ratio}_{loss}', kind=payload['kind'], config=relative.as_posix(), ready=False))
    (destination / 'STAGES.json').write_text(json.dumps(stages, indent=2)+'\n', encoding='utf-8')
    bindings = {name: '' for name in ('PRETRAIN_TRAIN_MANIFEST', 'PRETRAIN_VAL_MANIFEST', 'NEW_TRAIN_MANIFEST', 'NEW_VAL_MANIFEST', 'INIT_ROOT', 'SG_PRETRAIN_ROOT', 'WH_TEACHER_ROOT', 'RUN_ROOT')}
    (destination / 'bindings.example.json').write_text(json.dumps(bindings, indent=2)+'\n', encoding='utf-8')
    requirements = ['torch', 'torchvision', 'numpy', 'opencv-python', 'Pillow', 'PyYAML', 'scipy',
                    'scikit-image', 'lmdb', 'addict', 'future', 'requests', 'tqdm', 'tensorboard',
                    'yapf', 'kornia>=0.7,<0.9', 'editdistance', 'pytest']
    (destination / 'requirements_runtime.txt').write_text('\n'.join(requirements)+'\n', encoding='utf-8')
    versions = {}
    for package in requirements:
        package = package.split('>=')[0]
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = 'not installed locally'
    (destination / 'environment_reference.json').write_text(json.dumps(versions, indent=2)+'\n', encoding='utf-8')
    (destination / 'README.md').write_text(README, encoding='utf-8')
    (destination / 'README_CN.md').write_text(README_CN, encoding='utf-8')
    hashes = {path.relative_to(destination).as_posix(): digest(path) for path in sorted(destination.rglob('*')) if path.is_file()}
    (destination / 'MANIFEST.json').write_text(json.dumps(dict(stage_count=len(stages), supervised_templates=25,
        kd_components=8, datasets_included=False, learned_checkpoints_included=False, sha256=hashes), indent=2)+'\n', encoding='utf-8')
    assert len(stages) == 33
    return destination


README = '''# SG/WH data-free training kit

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
'''


README_CN = (ROOT / 'docs/training/sg_wh_training_kit_README_CN.md').read_text(encoding='utf-8')


def main():
    """@luna-comment
    Input: optional new kit folder name. Output: isolated directory and ZIP with external SHA256.
    Key Steps: assemble once, archive only kit files, verify no duplicate entries and record checksum.
    Key Params: default destination under delivery; existing destination/archive are rejected.
    State/Dependencies: source reads and new local output writes; no training/network actions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='sg_wh_training_kit_20261106')
    args = parser.parse_args()
    destination = ROOT / 'delivery' / args.name
    archive = destination.with_suffix('.zip')
    if archive.exists(): raise FileExistsError(archive)
    assemble(destination)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(destination.rglob('*')):
            if path.is_file(): output.write(path, (Path(destination.name) / path.relative_to(destination)).as_posix())
    archive.with_suffix('.zip.sha256').write_text(f'{digest(archive)}  {archive.name}\n', encoding='utf-8')
    print(json.dumps(dict(directory=str(destination), archive=str(archive), archive_bytes=archive.stat().st_size), indent=2))


if __name__ == '__main__':
    main()
