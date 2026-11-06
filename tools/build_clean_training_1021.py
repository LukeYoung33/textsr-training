"""Create the static training dependency closure and portable 1021 templates."""
import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'deployment/textsr_training_1021'
TASK = 'task/clean_training_project_1021_20261022.md'
STUBS = {'basicsr', 'realesrgan', 'basicsr.metrics', 'realesrgan.archs',
         'realesrgan.data', 'realesrgan.models', 'realesrgan.losses',
         'realesrgan.metrics', 'tools'}
SEEDS = ['basicsr.train', 'basicsr.archs', 'basicsr.data', 'basicsr.models',
         'basicsr.losses', 'basicsr.losses.basic_loss', 'basicsr.models.sr_model',
         'basicsr.metrics.psnr_ssim', 'realesrgan.archs.basic_text_line_r4_student_y4uv_v3_arch',
         'realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch',
         'realesrgan.losses.crnn_content_perceptual_loss',
         'realesrgan.losses.background_low_frequency_loss',
         'realesrgan.data.offline_augmented_region_dataset',
         'realesrgan.models.native_region_0921_model', 'realesrgan.models.block256_region_model']


def module_file(name):
    """@luna-comment
    Input: dotted local module. Output: existing Python path or None.
    Key Steps: resolve module then package initializer. Key Params: ROOT boundary.
    State/Dependencies: filesystem only; no import or code execution.
    """
    base = ROOT.joinpath(*name.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate


def closure():
    """@luna-comment
    Input: explicit runtime seeds. Output: complete local static import files.
    Key Steps: parse imports including relative symbols, traverse until stable.
    Key Params: excluded package initializers use explicit runtime registration.
    State/Dependencies: AST and repository source, never third-party directories.
    """
    pending, found, externals = list(SEEDS), {}, set()
    while pending:
        name = pending.pop()
        if name in found or name in STUBS:
            continue
        path = module_file(name)
        if path is None:
            continue
        found[name] = path
        for end in range(1, len(name.split('.'))):
            parent = '.'.join(name.split('.')[:end])
            if parent not in STUBS and module_file(parent):
                pending.append(parent)
        package = name.split('.') if path.name == '__init__.py' else name.split('.')[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                base = '.'.join([*prefix, *([node.module] if node.module else [])])
                names = [base, *[base + '.' + a.name for a in node.names]]
            for target in names:
                if target.split('.')[0] in ('basicsr', 'realesrgan', 'tools'):
                    if module_file(target):
                        pending.append(target)
                elif target:
                    externals.add(target.split('.')[0])
    # Windows source checkout stores DiffJPEG.py while imports use diffjpeg.
    # Avoid mistaking the exported DiffJPEG class for an additional module.
    found.pop('basicsr.utils.DiffJPEG', None)
    return found, sorted(externals)


def build():
    """@luna-comment
    Input: approved local 1021 baseline. Output: portable source/templates/provenance.
    Key Steps: copy unchanged import closure, emit isolated package roots and six configs.
    Key Params: no data/weights included; 250K/batch4 retained.
    State/Dependencies: writes only deployment project and task audit snapshot.
    """
    import yaml
    DEST.mkdir(parents=True, exist_ok=True)
    files, external = closure()
    inventory = []
    for name, source in sorted(files.items()):
        relative = source.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory.append(dict(path=relative.as_posix(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              read_range='whole_file_AST_import_inventory', behavior='unchanged'))
        for parent in target.parents:
            if parent == DEST:
                break
            init = parent / '__init__.py'
            if not init.exists():
                init.write_text('"""Explicitly registered portable training package."""\n', encoding='utf-8')
    for name in STUBS:
        path = DEST.joinpath(*name.split('.')) / '__init__.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Explicitly registered portable training package."""\n', encoding='utf-8')
    for item in (DEST / 'basicsr/utils').iterdir():
        if item.name == 'DiffJPEG.py':
            temporary = item.with_name('diffjpeg_case_fix.tmp')
            item.rename(temporary)
            temporary.rename(item.with_name('diffjpeg.py'))
    source = ROOT / 'evaluation/offline_v3_v5_real_synthetic_training_20261020/configs/02_v3_real.yml'
    base = yaml.safe_load(source.read_text())
    v5_source = ROOT / 'options/train_bucket_3/lenovo_v5_p17_effect_iteration_20260920/train_original_r1_bglf_uv_250k_20260920.yml'
    v5 = yaml.safe_load(v5_source.read_text())
    for mode, preprocessing in [('text_line', 'none'), ('text_line', 'both'), ('block256', 'none')]:
        for architecture in ('v3', 'v5'):
            config = copy.deepcopy(base)
            hw = [256, 256] if mode == 'block256' else [96, 480]
            config['name'] = f'{mode}_{architecture}_{preprocessing}_1021'
            config['project'] = dict(mode=mode, preprocessing=preprocessing, baseline='2026-10-21')
            config['network_g'] = copy.deepcopy(v5['network_g'] if architecture == 'v5' else base['network_g'])
            config['network_g']['canvas_hw'] = hw
            config['model_type'] = 'Block256RegionModel' if mode == 'block256' else 'NativeRegion0921Model'
            config['train']['model_type'] = config['model_type']
            config['train']['cp_opt']['recognizer_path'] = 'weights/scene_base_CRNN.pth'
            config['train']['cp_opt']['alphabet_path'] = 'realesrgan/metrics/benchmark.txt'
            if mode == 'block256':
                config['train']['cp_opt']['type'] = 'CRNNTextLineContentLoss'
            config['path'] = dict(pretrain_network_g=f'weights/{architecture}_init.pth',
                                  strict_load_g=True, resume_state=None, experiments_root='experiments')
            # Synthetic-degradation settings are never consumed by offline reader/feed.
            for key in list(config):
                if key.startswith(('resize_', 'gaussian_noise_', 'noise_range', 'poisson_',
                                   'gray_noise_', 'jpeg_range', 'second_blur_')):
                    config.pop(key)
            bank = f'data/{mode}_{preprocessing}'
            for split, dataset in config['datasets'].items():
                keep = {'type', 'batch_size_per_gpu', 'num_worker_per_gpu', 'dataset_enlarge_ratio',
                        'use_shuffle', 'cache_shards'}
                config['datasets'][split] = {k: v for k, v in dataset.items() if k in keep}
                config['datasets'][split].update(name=f'{mode}_{preprocessing}_{split}', split=split,
                    canvas_hw=hw, root=bank,
                    manifest=f'{bank}/' + ('train_balanced.json' if split == 'train' else 'val.json'))
            target = DEST / 'configs' / (config['name'] + '.yml')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    for relative in ('realesrgan/metrics/benchmark.txt', 'LICENSE'):
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    report = dict(task=TASK, baseline='2026-10-21',
        configuration_sources=[str(source.relative_to(ROOT)), str(v5_source.relative_to(ROOT))],
        configuration_note='Reconstructs 1021 manager.prepare recipe from retained baseline configs; original remote 1021 YAMLs not accessed.',
        source_files=inventory, external_imports=external,
        omitted='datasets, weights, historical tools, remote orchestration, caches, logs')
    (DEST / 'PROVENANCE.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    baselines = {}
    for label, relative in [('text_line', 'evaluation/vosr_augmented_training_20261021/provenance.json'),
                            ('block256', 'evaluation/vosr_block256_training_20261021/provenance.json')]:
        recorded = json.loads((ROOT / relative).read_text())
        baselines[label] = {arm: dict(checkpoint_sha256=entry['checkpoint_sha256'],
            original_checkpoint=entry['checkpoint'], original_config_sha256=entry['config_sha256'])
            for arm, entry in recorded.items()}
    (DEST / 'BASELINE_ASSETS.json').write_text(json.dumps(dict(
        description='Recorded 1021 warm starts. These are initial weights, not final VOSR weights.',
        portable_names=dict(v3='weights/v3_init.pth', v5='weights/v5_init.pth',
                            crnn='weights/scene_base_CRNN.pth'), experiments=baselines), indent=2), encoding='utf-8')
    (ROOT / 'task/clean_training_project_1021_20261022.read_audit.json').write_text(
        json.dumps(dict(task_ref=TASK, last_updated_at='2026-10-22',
            read_purpose='Static import closure; unchanged copies rather than function-call review.',
            read_set=inventory), indent=2), encoding='utf-8')
    print(json.dumps(dict(source_files=len(files), external_imports=external)))


def archive():
    """@luna-comment
    Input: completed portable project. Output: source ZIP with per-file hash manifest.
    Key Steps: whitelist source/config/license assets, hash files, compress and CRC-check.
    Key Params: exclude caches/runs/data/weights; archive stays inside repository.
    State/Dependencies: only project source and owned release archive are written.
    """
    paths = sorted(p for p in DEST.rglob('*') if p.is_file()
        and '__pycache__' not in p.relative_to(DEST).parts
        and p.relative_to(DEST).parts[0] not in {'runs', 'data', 'weights', 'experiments', 'smoke_output'}
        and (p.suffix in {'.py', '.yml', '.yaml', '.json', '.md', '.txt'} or p.name in {'LICENSE', '.gitignore'})
        and p.name != 'PACKAGE_MANIFEST.json')
    manifest = {p.relative_to(DEST).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest_path = DEST / 'PACKAGE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    output = ROOT / 'deployment/TextSR_Training_1021_20261022.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in [*paths, manifest_path]:
            bundle.write(path, 'textsr_training_1021/' + path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(output) as bundle:
        assert bundle.testzip() is None
    print(json.dumps(dict(archive=str(output.relative_to(ROOT)), files=len(paths)+1,
                         bytes=output.stat().st_size, sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    if args.archive_only:
        archive()
    else:
        build()
