"""Read-only historical training data/config/provenance inventory for user handoff."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import yaml

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
EXP = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments')
OUT = EXP.parent/'evaluation/training_dataset_handoff_20261020'
REPO = Path(__file__).resolve().parents[1]
CORE = {'lq', 'gt', 'ocr', 'text_mask', 'valid_mask'}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.json', '.jsonl', '.txt', '.npz', '.npy'}


def sha(path):
    """@luna-comment Input:source file. Output:SHA256. Key Steps:stream1MiB chunks.
    Key Params:encoded source bytes. State/Dependencies:read-only file access.
    """
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):h.update(block)
    return h.hexdigest()


def windows(path):
    """@luna-comment Input:absolute Linux asset path. Output:verified mount-prefix Y path or empty.
    Key Steps:map only established /AI_DNS prefix. Key Params:no guessed /home or /data aliases.
    State/Dependencies:existing shared mount convention, no filesystem mutation.
    """
    text = str(path)
    return 'Y:\\'+text[len('/AI_DNS/'):].replace('/', '\\') if text.startswith('/AI_DNS/') else ''


def path_fields(obj, prefix=''):
    """@luna-comment Input:JSON metadata tree. Output:role/path pairs for declared file assets.
    Key Steps:recursive traversal, file-extension filter and identity-field exclusion.
    Key Params:known image/annotation/array suffixes; no inferred filenames.
    State/Dependencies:read-only source object, yields directory-free declarations.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in {'sample_id', 'source_id', 'name', 'text', 'source_identity', 'input_hashes', 'output_sha256', 'source_hashes'}:
                yield from path_fields(value, prefix+'.'+key if prefix else key)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):yield from path_fields(value, prefix+f'[{index}]')
    elif isinstance(obj, str) and Path(obj).suffix.lower() in EXTENSIONS:
        yield prefix, obj


class Inventory:
    """Task-owned metadata accumulator; never writes into source data directories."""

    def __init__(self):
        """@luna-comment Input:fixed output path. Output:empty dataset/model/file registries.
        Key Steps:prepare own directories and caches. Key Params:stable dataset insertion IDs.
        State/Dependencies:source directories remain read-only throughout collection.
        """
        OUT.mkdir(parents=True, exist_ok=True)
        for folder in ['manifests', 'configs', 'metadata']:(OUT/folder).mkdir(exist_ok=True)
        self.datasets = [];self.models = [];self.bindings = [];self.warnings = []
        self.dataset_keys = {};self.directories = {};self.assets = {};self.pairs = {}
        self.metrics_seen = set();self.lineage_count = 0

    def exists(self, path):
        """@luna-comment Input:declared file path. Output:current directory-entry existence.
        Key Steps:cache each parent directory listing, handle unavailable parents explicitly.
        Key Params:no full image hashes/stat per repeated reference. State/Dependencies:read-only cache.
        """
        p = Path(path);parent = str(p.parent)
        if parent not in self.directories:
            try:self.directories[parent] = {e.name for e in os.scandir(parent)}
            except OSError:self.directories[parent] = set()
        return p.name in self.directories[parent]

    def asset(self, path, dataset, role, level='provenance'):
        """@luna-comment Input:absolute declared asset, dataset ID and provenance role.
        Output:existence flag and deduplicated file-registry entry.
        Key Steps:directory check, accumulate role/dataset/criticality without reading images.
        Key Params:only actual primary train inputs use core level. State/Dependencies:own metadata state.
        """
        path = str(path)
        if path not in self.assets:
            self.assets[path] = dict(exists=self.exists(path), datasets=set(), roles=set(), levels=set())
        item = self.assets[path];item['datasets'].add(dataset);item['roles'].add(role);item['levels'].add(level)
        return item['exists']

    def add_dataset(self, root, manifest, split, family, stage, used_by=None):
        """@luna-comment Input:actual dataset root/manifest/split and model binding.
        Output:stable dataset ID, exact source paths and live manifest metadata.
        Key Steps:resolve declared paths, deduplicate identical bindings and record aliases/use.
        Key Params:no manifest-content reinterpretation yet. State/Dependencies:source metadata only.
        """
        root = Path(root);manifest = Path(manifest)
        if not manifest.is_absolute():
            choices = [root/manifest, root/'manifests'/manifest]
            manifest = next((p for p in choices if p.exists()), choices[0])
        key = (str(root), str(manifest), split)
        if key not in self.dataset_keys:
            ident = f'D{len(self.datasets)+1:03d}';self.dataset_keys[key] = ident
            self.datasets.append(dict(id=ident, family=family, stage=stage, split=split,
                root_linux=str(root), root_windows=windows(root), manifest_linux=str(manifest),
                manifest_windows=windows(manifest), manifest_exists=manifest.is_file(),
                manifest_bytes=manifest.stat().st_size if manifest.is_file() else 0, used_by=[], samples=0, patches=0))
        ident = self.dataset_keys[key]
        if used_by and used_by not in self.datasets[int(ident[1:])-1]['used_by']:
            self.datasets[int(ident[1:])-1]['used_by'].append(used_by)
        return ident

    def config(self, family, config, run_root):
        """@luna-comment Input:actual historical training YAML and experiment directory.
        Output:model record, copied original config and exact train/val source bindings.
        Key Steps:safe parse; record metadata-only trailing-port repair; discover source
        roots/manifests and artifact presence; preserve original YAML bytes unchanged.
        Key Params:source fields preserved; no trainer execution or model loading.
        State/Dependencies:read-only experiments, task-owned snapshots and mapping rows.
        """
        text = config.read_text(encoding='utf-8');repair = ''
        try:options = yaml.safe_load(text)
        except yaml.YAMLError:
            repaired = re.sub(r'^\s*:\s*\d+\s*$', '', text, flags=re.M)
            try:
                options = yaml.safe_load(repaired);repair = 'ignored malformed standalone trailing port line for metadata parse only'
                self.warnings.append(dict(kind='config_parse_repair', path=str(config), detail=repair))
            except yaml.YAMLError as error:
                self.warnings.append(dict(kind='unparsed_config', path=str(config), detail=str(error)));return
        if not isinstance(options, dict) or not options.get('datasets'):return
        mid = f'M{len(self.models)+1:03d}'
        snapshot = OUT/'configs'/(mid+'.yml');shutil.copy2(config, snapshot)
        path_settings = options.get('path') or {};configured_run = path_settings.get('experiments_root')
        candidates = [Path(configured_run)] if configured_run else []
        if configured_run:candidates.append(Path(configured_run)/str(options.get('name', '')))
        candidates += [Path(run_root), Path(run_root)/str(options.get('name', ''))]
        weights = []
        for candidate in candidates:
            weights += [str(p) for p in (candidate/'models').glob('net_g*.pth')]
        policy = 'offline paired input'
        if family == 'v3_native_20261015':
            name = str(options.get('name', config.stem))
            policy = 'paired75% + online_GT_degradation25%' if 'mixed75' in name else ('online_GT_degradation' if 'degraded' in name else 'paired100%')
        elif options.get('high_order_degradation'):policy = 'online_GT_degradation'
        entry = dict(id=mid, family=family, name=options.get('name', config.stem), config_linux=str(config),
                     config_windows=windows(config), config_sha256=sha(config), config_snapshot=str(snapshot.relative_to(OUT)),
                     run_root_linux=str(run_root), run_root_windows=windows(run_root), observed_generator_files=sorted(set(weights)),
                     run_artifacts_present=bool(weights), parse_note=repair, model_type=options.get('model_type'),
                     network_type=(options.get('network_g') or {}).get('type'), input_policy=policy,
                     high_order_degradation=options.get('high_order_degradation'),
                     initialization_linux=path_settings.get('pretrain_network_g', path_settings.get('pretrain_net_g', '')),
                     data_options=options['datasets'], dataset_ids=[])
        self.models.append(entry)
        for split, dataset in options['datasets'].items():
            if not isinstance(dataset, dict):continue
            sources = dataset.get('sources') or [dataset]
            for source in sources:
                root = source.get('root') or dataset.get('root') or source.get('dataroot') or dataset.get('dataroot')
                manifest = source.get('manifest') or dataset.get('manifest')
                if not manifest:
                    self.warnings.append(dict(kind='dataset_manifest_not_declared', model=mid, split=split, detail=dataset));continue
                root = root or str(Path(manifest).parent)
                stage = 'native_patch_index' if family == 'v3_native_20261015' else ('runtime_random_fullframe_patches' if family == 'fullframe_20261014' else 'training_manifest')
                ident = self.add_dataset(root, manifest, split, family, stage, mid);entry['dataset_ids'].append(ident)
                self.bindings.append(dict(model_id=mid, model_name=entry['name'], family=family, split=split, dataset_id=ident,
                    canvas_hw=json.dumps(dataset.get('canvas_hw')), input_policy=policy,
                    source_name=source.get('name', ''), weight=source.get('weight', ''), epoch_size=dataset.get('epoch_size', ''),
                    config_linux=str(config), config_windows=windows(config), manifest_declared=str(manifest),
                    source_manifest=dataset.get('source_manifest', ''), resize_mode=dataset.get('resize_mode', '')))
                for key in ['source_manifest', 'source_manifest_original', 'derived_source_manifest']:
                    extra = dataset.get(key)
                    if extra:self.add_dataset(Path(extra).parent, extra, split, family+'_source', key, mid)

    def discover(self):
        """@luna-comment Input:known conversation experiment families.
        Output:live model/config discovery plus published and upstream manifest coverage.
        Key Steps:exclude archived diagnostic runs, retain observed/configured distinction,
        add existing raw-pair and published variants without claiming they were trained.
        Key Params:bounded named roots, no broad filesystem search. State/Dependencies:read-only discovery.
        """
        initial = EXP/'basic_text_line_y4uv_v3_cp_haar_20260820_96x480'
        for p in initial.glob('*.yml'):self.config('v3_initial_20260820', p, initial)
        old = EXP/'lenovo_v5_p17_effect_iteration_20260920/lenovo_v5_p17_original_r1_bglf_uv_250k_20260920'
        for p in old.glob('*.yml'):self.config('0921_original', p, old)
        for group, base in [('0928_main', EXP/'lenovo_borderrail_8arm_20260928'), ('0928_finetune', EXP/'lenovo_borderrail_8arm_mojito_finetune_20260928')]:
            for directory in sorted(base.iterdir()):
                if not directory.is_dir() or '_archived_' in directory.name:continue
                for p in directory.glob('*.yml'):self.config(group, p, directory)
        for family, base in [('v5_four_20261014', EXP/'displayfit_4arm_20261014'), ('v3_native_20261015', EXP/'v3_native_four_20261015')]:
            for p in sorted((base/'configs').glob('*.yml')):self.config(family, p, base/'runs'/p.stem)
        base = EXP/'fullframe_four_data_20261014'
        for p in sorted((base/'configs').glob('*/train.yml')):self.config('fullframe_20261014', p, base/'runs'/p.parent.name)
        for date in ['20261014', '20261015']:
            for variant in ['full_images', 'text_lines', 'text_lines_preprocessed', 'text_lines_ppocrv6_medium']:
                root = DATA/f'displayfit_training_{date}'/variant
                for manifest in sorted((root/'manifests').glob('*.jsonl')):
                    self.add_dataset(root, manifest, manifest.stem, 'displayfit_'+date, variant)
        old_root = Path('/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop')
        for p in sorted((old_root/'manifests').glob('*.jsonl')):
            stage='selection_decisions_metadata' if p.stem=='selection_decisions' else 'valid_cropped_parent_images'
            self.add_dataset(old_root, p, p.stem, '0921_source', stage)
        for root in [DATA/'lenovo_data_26101/alignment_20261012', DATA/'HX_data_1013/alignment_20261013/A', DATA/'HX_data_1013/alignment_20261013/B']:
            p = root/'pairs.jsonl'
            if p.exists():
                self.add_dataset(root, p, 'source_mapping', 'raw_camera_pairs', 'raw_source_pairs')
                with p.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row.get('pair_id'):self.pairs[row['pair_id']] = row

    def lineage(self, row, dataset, writer):
        """@luna-comment Input:published parent or historical source provenance record.
        Output:upstream full-resolution/reference/geometry/crop linkage rows.
        Key Steps:follow explicit displayfit metrics or old status JSON, join original
        raw pair by sample identity, retain uncropped aligned sibling LQ when available.
        Key Params:no fabricated original-photo correspondence; missing paths retained.
        State/Dependencies:bounded source metadata reads and task-owned CSV writer.
        """
        prov = row.get('source_provenance') or {};sid = row.get('sample_id', '')
        metric = prov.get('metadata')
        old_photo = (prov.get('source_record') or {}).get('new_photometric') or {}
        if not metric and old_photo:metric = old_photo.get('status_file')
        if not metric or metric in self.metrics_seen:return
        self.metrics_seen.add(metric)
        entry = dict(dataset_id=dataset['id'], sample_id=sid, source_identity=row.get('source_identity', ''),
                     split=dataset['split'], metadata_linux=metric, metadata_windows=windows(metric), crop_xywh=json.dumps(row.get('crop_xywh')))
        if self.exists(metric):
            meta = json.loads(Path(metric).read_text());inputs = meta.get('inputs') or {}
            for role in ['display', 'spatial_gt', 'geometry', 'lq', 'text_mask', 'valid_mask', 'ocr']:
                value = inputs.get(role)
                if value:
                    entry[role+'_linux'] = value;entry[role+'_windows'] = windows(value)
                    self.asset(value, dataset['id'], 'upstream.'+role)
            entry['input_lq_cropped'] = inputs.get('lq_cropped', '')
            gt = inputs.get('spatial_gt')
            if gt:
                for name in ['lq.jpg', 'lq.png', 'LQ.jpg', 'LQ.png']:
                    candidate = Path(gt).parent/name
                    if self.exists(candidate):
                        entry['aligned_uncropped_lq_linux'] = str(candidate);entry['aligned_uncropped_lq_windows'] = windows(candidate)
                        self.asset(candidate, dataset['id'], 'upstream.aligned_uncropped_lq');break
        pair = self.pairs.get(sid)
        if pair:
            for role in ['lq', 'gt']:
                if pair.get(role):entry['raw_pair_'+role+'_linux'] = pair[role];entry['raw_pair_'+role+'_windows'] = windows(pair[role])
        writer.write(json.dumps(entry, ensure_ascii=True)+'\n');self.lineage_count += 1

    def collect(self):
        """@luna-comment Input:discovered immutable dataset/model registry.
        Output:full manifest/sample/patch/file/lineage inventories and live verification.
        Key Steps:stream each source manifest with SHA snapshot; normalize paths,
        enumerate exact native coordinates, check declared assets via parent listings.
        Key Params:all splits including excluded, no image duplication or byte-hash claim.
        State/Dependencies:read-only data, own compressed metadata and progress files.
        """
        fields = ['dataset_id', 'split', 'sample_id', 'source_identity', 'parent_sample_id', 'record_status', 'patch_count',
                  'lq_linux', 'lq_windows', 'gt_linux', 'gt_windows', 'ocr_linux', 'ocr_windows',
                  'text_mask_linux', 'text_mask_windows', 'valid_mask_linux', 'valid_mask_windows',
                  'source_manifest_linux', 'crop_xywh', 'expected_hashes_json', 'extra_assets_json']
        with gzip.open(OUT/'samples.csv.gz', 'wt', encoding='utf-8', newline='') as sample_file, \
             gzip.open(OUT/'native_patches.csv.gz', 'wt', encoding='utf-8', newline='') as patch_file, \
             gzip.open(OUT/'upstream_lineage.jsonl.gz', 'wt', encoding='utf-8') as lineage:
            sample_writer = csv.DictWriter(sample_file, fieldnames=fields);sample_writer.writeheader()
            patch_writer = csv.writer(patch_file);patch_writer.writerow(['dataset_id', 'split', 'sample_id', 'patch_index', 'x', 'y', 'width', 'height', 'source_lq_linux', 'source_gt_linux'])
            for dataset in self.datasets:
                path = Path(dataset['manifest_linux']);root = Path(dataset['root_linux'])
                if not path.is_file():
                    dataset['error'] = 'manifest_missing';self.warnings.append(dict(kind='missing_manifest', dataset=dataset['id'], path=str(path)));continue
                digest = hashlib.sha256();families = Counter();ids = set();duplicates = 0;core_missing = 0
                snapshot = OUT/'manifests'/(dataset['id']+'.jsonl.gz');dataset['snapshot'] = str(snapshot.relative_to(OUT))
                with path.open('rb') as stream, gzip.open(snapshot, 'wb') as target:
                    for number, raw in enumerate(stream, 1):
                        digest.update(raw);target.write(raw)
                        if not raw.strip():continue
                        record = json.loads(raw);sid = str(record.get('sample_id', record.get('source_id', record.get('pair_id', record.get('id', number)))))
                        duplicates += sid in ids;ids.add(sid)
                        files = {k:v for k,v in (record.get('files') or {}).items() if isinstance(v,str)} if isinstance(record.get('files'), dict) else {}
                        for key in CORE:
                            if isinstance(record.get(key), str):files[key] = record[key]
                            if isinstance(record.get(key+'_path'), str):files[key] = record[key+'_path']
                        resolved = {key:str(Path(value) if Path(value).is_absolute() else root/value) for key,value in files.items()}
                        extras = []
                        for role, value in path_fields(record):
                            asset_path = str(Path(value) if Path(value).is_absolute() else root/value)
                            leaf = role.split('.')[-1]
                            level = 'core' if (role.startswith('files.') and leaf in CORE) or role in CORE or role in {k+'_path' for k in CORE} else 'provenance'
                            exists = self.asset(asset_path, dataset['id'], role, level)
                            if level == 'core' and not exists:core_missing += 1
                            if level != 'core':extras.append(dict(role=role, linux=asset_path, windows=windows(asset_path), exists=exists))
                        patches = record.get('patches') or []
                        output = dict(dataset_id=dataset['id'], split=dataset['split'], sample_id=sid,
                            source_identity=record.get('source_identity', record.get('capture_group', '')),
                            parent_sample_id=record.get('parent_sample_id', ''), record_status=record.get('state', record.get('status', '')),
                            patch_count=len(patches), source_manifest_linux=str(path), crop_xywh=json.dumps(record.get('crop_xywh', record.get('line_envelope_xyxy_in_parent'))),
                            expected_hashes_json=json.dumps(record.get('output_sha256', record.get('source_hashes', {}))), extra_assets_json=json.dumps(extras))
                        for role in CORE:output[role+'_linux'] = resolved.get(role, '');output[role+'_windows'] = windows(resolved[role]) if role in resolved else ''
                        sample_writer.writerow(output)
                        for index, patch in enumerate(patches):
                            xy = patch['xy'] if isinstance(patch, dict) else patch
                            patch_writer.writerow([dataset['id'], dataset['split'], sid, index, xy[0], xy[1], 480, 96, resolved.get('lq', ''), resolved.get('gt', '')])
                        dataset['samples'] += 1;dataset['patches'] += len(patches)
                        families[str(record.get('dataset', record.get('source', 'unspecified')))] += 1
                        if dataset['stage'] in ['full_images', 'valid_cropped_parent_images']:self.lineage(record, dataset, lineage)
                dataset.update(manifest_sha256=digest.hexdigest(), source_counts=dict(families), duplicate_sample_ids=duplicates, core_missing_references=core_missing)
                (OUT/'progress.json').write_text(json.dumps(dict(dataset=dataset['id'], completed=self.datasets.index(dataset)+1, total=len(self.datasets), assets=len(self.assets))))
                print('DATASET', dataset['id'], dataset['stage'], dataset['split'], dataset['samples'], 'patches', dataset['patches'], 'missing_core', core_missing, flush=True)

    def finish(self):
        """@luna-comment Input:complete metadata accumulators.
        Output:CSV/JSON catalogs, referenced/missing file lists and acceptance summary.
        Key Steps:deduplicate assets, retain missing/remote-only states, serialize models
        and split bindings, publish counts without summing duplicate data versions.
        Key Params:existence inventory is not a fresh all-image checksum validation.
        State/Dependencies:task output only; no source files modified or linked.
        """
        with gzip.open(OUT/'referenced_files.csv.gz', 'wt', encoding='utf-8', newline='') as file, \
             (OUT/'missing_files.csv').open('w', encoding='utf-8-sig', newline='') as missing:
            columns = ['linux_path', 'windows_path', 'exists', 'dataset_ids', 'roles', 'levels']
            writer = csv.DictWriter(file, fieldnames=columns);writer.writeheader()
            missing_writer = csv.DictWriter(missing, fieldnames=columns);missing_writer.writeheader()
            for path, info in sorted(self.assets.items()):
                item = dict(linux_path=path, windows_path=windows(path), exists=info['exists'], dataset_ids=';'.join(sorted(info['datasets'])),
                            roles=';'.join(sorted(info['roles'])), levels=';'.join(sorted(info['levels'])))
                writer.writerow(item)
                if not info['exists']:missing_writer.writerow(item)
        for name, rows in [('datasets', self.datasets), ('models', self.models), ('model_dataset_bindings', self.bindings)]:
            (OUT/(name+'.json')).write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding='utf-8')
            keys = sorted({k for row in rows for k in row})
            with (OUT/(name+'.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys);writer.writeheader()
                for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        (OUT/'warnings.json').write_text(json.dumps(self.warnings, indent=2, ensure_ascii=True))
        report = dict(status='collected', date='2026-10-20', models=len(self.models), model_bindings=len(self.bindings), datasets=len(self.datasets),
                      sample_rows=sum(d['samples'] for d in self.datasets), native_patch_rows=sum(d['patches'] for d in self.datasets),
                      unique_declared_files=len(self.assets), existing_files=sum(v['exists'] for v in self.assets.values()),
                      missing_files=sum(not v['exists'] for v in self.assets.values()),
                      missing_core_files=sum(not v['exists'] and 'core' in v['levels'] for v in self.assets.values()),
                      lineage_rows=self.lineage_count, warnings=len(self.warnings),
                      verification_scope='all declared path existence and manifest/config SHA256; image hashes are published metadata only')
        (OUT/'collection.json').write_text(json.dumps(report, indent=2));print(json.dumps(report), flush=True)


def main():
    """@luna-comment Input:fixed historical experiment roots. Output:complete handoff metadata.
    Key Steps:discover -> collect -> finish. Key Params:no training/data mutation.
    State/Dependencies:read-only Remote58 datasets and isolated versioned output.
    """
    started = time.perf_counter();inventory = Inventory();inventory.discover();inventory.collect();inventory.finish()
    print('ELAPSED', round(time.perf_counter()-started,2), flush=True)


if __name__ == '__main__':main()
