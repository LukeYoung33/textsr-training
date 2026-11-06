"""Resolve actual run directories, retain release evidence and validate handoff metadata."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
from collections import Counter
import yaml

from build_training_dataset_handoff_20261020 import OUT, REPO, DATA, EXP, sha, windows, path_fields, CORE, Inventory


def write_catalog(name, rows):
    """@luna-comment Input:catalog rows. Output:consistent JSON and UTF8-BOM CSV.
    Key Steps:union keys and encode nested metadata. Key Params:exact strings retained.
    State/Dependencies:task-owned catalog only.
    """
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({key for row in rows for key in row}));writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(value) if isinstance(value,(dict,list)) else value for key,value in row.items()})


def main():
    """@luna-comment Input:completed manifest collector and raw-root inventory.
    Output:correct run/model bindings, release evidence, counts/hash validation.
    Key Steps:resolve configured base/name directories, preserve upstream metadata,
    stream source snapshots and normalized rows, check patch totals and bindings.
    Key Params:no source image mutation/hash recomputation; explicit failure evidence.
    State/Dependencies:source metadata reads and isolated handoff writes.
    """
    models=json.loads((OUT/'models.json').read_text());datasets=json.loads((OUT/'datasets.json').read_text());checks=[]
    extra_inventory=Inventory();base_lineage=0
    with gzip.open(OUT/'upstream_lineage.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                entry=json.loads(line);base_lineage+=1;extra_inventory.metrics_seen.add(entry.get('metadata_linux'))
    old_ids={ident for model in models if model['family']=='0921_original' for ident in model['dataset_ids']}
    for model in models:
        raw=(OUT/model['config_snapshot']).read_text()
        options=yaml.safe_load(re.sub(r'^\s*:\s*\d+\s*$','',raw,flags=re.M))
        model['runtime_lq_preprocess']=options.get('lq_preprocess') or {}
        model['total_iter']=(options.get('train') or {}).get('total_iter')
        base=(options.get('path') or {}).get('experiments_root');candidates=[]
        if base:candidates.extend([Path(base)/model['name'],Path(base)])
        candidates.extend([Path(model['run_root_linux']),Path(model['config_linux']).parent])
        hits={str(path) for directory in candidates for path in (directory/'models').glob('net_g*.pth')}
        model['observed_generator_files']=sorted(hits);model['run_artifacts_present']=bool(hits)
        model['resolved_run_directories']=sorted({str(Path(path).parent.parent) for path in hits})
        checks.append(dict(check='config_snapshot_sha256',id=model['id'],passed=sha(OUT/model['config_snapshot'])==model['config_sha256']))
    write_catalog('models',models)
    evidence=[]
    for date in ['20261014','20261015']:
        base=DATA/f'displayfit_training_{date}'
        for folder in [base,base/'text_lines_preprocessed']:
            for name in ['release_report.json','publish_summary.json','resolved_config.json','resolved_config.yaml','preprocess_summary.json','summary.json','freeze_manifest.json']:
                path=folder/name
                if path.is_file():
                    target=OUT/'metadata'/(date+'_'+folder.name+'_'+name);shutil.copy2(path,target)
                    evidence.append(dict(source_linux=str(path),source_windows=windows(path),snapshot=str(target.relative_to(OUT)),sha256=sha(path)))
    freeze=REPO/'evaluation/edge_overlap_peak_validation_20261013/frozen_strict_combined_v1'
    for name in ['TECHNICAL_DESIGN.md','freeze_manifest.json']:
        path=freeze/name
        if path.is_file():
            target=OUT/'metadata'/('strict_combined_v1_'+name);shutil.copy2(path,target)
            evidence.append(dict(source_linux=str(path),snapshot=str(target.relative_to(OUT)),sha256=sha(path)))
    for path in [EXP/'v3_native_four_20261015/preparation.json', EXP/'displayfit_4arm_20261014/status.json',EXP/'fullframe_four_data_20261014/status.json']:
        if path.is_file():
            target=OUT/'metadata'/(path.parent.name+'_'+path.name);shutil.copy2(path,target)
            evidence.append(dict(source_linux=str(path),source_windows=windows(path),snapshot=str(target.relative_to(OUT)),sha256=sha(path)))
    imported=OUT/'local_source_metadata.json'
    if imported.is_file():evidence.extend(json.loads(imported.read_text(encoding='utf-8-sig')))
    for entry in evidence:
        checks.append(dict(check='release_metadata_snapshot_sha256',id=entry['snapshot'],passed=sha(OUT/entry['snapshot'])==entry['sha256']))
    (OUT/'metadata_sources.json').write_text(json.dumps(evidence,indent=2))
    ambiguous=set();unresolved_rows=0;row_updates={}
    with gzip.open(OUT/'unresolved_relative_references.csv.gz','wt',encoding='utf-8',newline='') as target, \
         gzip.open(OUT/'upstream_lineage.jsonl.gz','at',encoding='utf-8') as lineage_writer:
        writer=csv.writer(target);writer.writerow(['dataset_id','manifest_row','role','declared_relative_path','source_manifest','reason'])
        for d in datasets:
            if not d.get('snapshot'):continue
            digest=hashlib.sha256();count=0
            with gzip.open(OUT/d['snapshot'],'rb') as stream:
                for line in stream:
                    digest.update(line)
                    if not line.strip():continue
                    count+=1
                    record=json.loads(line)
                    if d['id'] in old_ids:extra_inventory.lineage(record,d,lineage_writer)
                    if not any(record.get(key) for key in ['sample_id','source_id','pair_id','id']):
                        identity=record.get('segment_id') or record.get('stem') or record.get('frame_id')
                        if identity:
                            row_updates[(d['id'],count)]={'sample_id':str(identity),
                                'source_identity':str(record.get('source_identity') or record.get('frame_id') or ''),
                                'parent_sample_id':str(record.get('parent_sample_id') or record.get('frame_id') or '')}
                    for role,value in path_fields(record):
                        root_bound=role.startswith('files.') or role in CORE or role in {k+'_path' for k in CORE}
                        if not Path(value).is_absolute() and not root_bound:
                            ambiguous.add(str(Path(d['root_linux'])/value));unresolved_rows+=1
                            writer.writerow([d['id'],count,role,value,d['manifest_linux'],'inherited provenance base directory not declared; not classified as missing'])
            checks.append(dict(check='manifest_snapshot_count_and_sha256',id=d['id'],passed=count==d['samples'] and digest.hexdigest()==d['manifest_sha256']))
    removed=set();missing_count=0;existing_count=0;known_count=0
    with gzip.open(OUT/'referenced_files.csv.gz','rt',encoding='utf-8',newline='') as source, \
         gzip.open(OUT/'referenced_files.resolved.csv.gz','wt',encoding='utf-8',newline='') as target, \
         (OUT/'missing_files.csv').open('w',encoding='utf-8-sig',newline='') as missing:
        reader=csv.DictReader(source);writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader()
        missing_writer=csv.DictWriter(missing,fieldnames=reader.fieldnames);missing_writer.writeheader()
        for row in reader:
            if row['linux_path'] in ambiguous and 'core' not in row['levels'].split(';'):
                removed.add(row['linux_path']);continue
            extra=extra_inventory.assets.pop(row['linux_path'],None)
            if extra:
                row['dataset_ids']=';'.join(sorted(set(row['dataset_ids'].split(';'))|extra['datasets']))
                row['roles']=';'.join(sorted(set(row['roles'].split(';'))|extra['roles']))
            writer.writerow(row);known_count+=1
            if row['exists']=='True':existing_count+=1
            else:missing_writer.writerow(row);missing_count+=1
        for path,info in sorted(extra_inventory.assets.items()):
            row=dict(linux_path=path,windows_path=windows(path),exists=info['exists'],dataset_ids=';'.join(sorted(info['datasets'])),roles=';'.join(sorted(info['roles'])),levels=';'.join(sorted(info['levels'])))
            writer.writerow(row);known_count+=1
            if info['exists']:existing_count+=1
            else:missing_writer.writerow(row);missing_count+=1
    (OUT/'referenced_files.resolved.csv.gz').replace(OUT/'referenced_files.csv.gz')
    actual=Counter()
    with gzip.open(OUT/'samples.csv.gz','rt',encoding='utf-8',newline='') as stream, \
         gzip.open(OUT/'samples.resolved.csv.gz','wt',encoding='utf-8',newline='') as target:
        reader=csv.DictReader(stream);writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader()
        for row in reader:
            actual[row['dataset_id']]+=1
            row.update(row_updates.get((row['dataset_id'],actual[row['dataset_id']]),{}))
            extras=json.loads(row['extra_assets_json'])
            for extra in extras:
                if extra.get('linux') in removed:
                    extra['unresolved_candidate_linux']=extra.pop('linux');extra['windows']='';extra['exists']=None;extra['resolution']='unresolved_relative_provenance; see exact source manifest'
            row['extra_assets_json']=json.dumps(extras);writer.writerow(row)
    (OUT/'samples.resolved.csv.gz').replace(OUT/'samples.csv.gz')
    collection=json.loads((OUT/'collection.json').read_text())
    collection.update(unique_declared_files=known_count,existing_files=existing_count,missing_files=missing_count,
                      lineage_rows=base_lineage+extra_inventory.lineage_count,
                      unique_training_runs=len({(m['family'],m['name']) for m in models}),
                      unresolved_relative_provenance_rows=unresolved_rows,
                      unresolved_relative_provenance_candidate_paths=max(len(removed),collection.get('unresolved_relative_provenance_candidate_paths',0)))
    (OUT/'collection.json').write_text(json.dumps(collection,indent=2))
    patches=Counter()
    with gzip.open(OUT/'native_patches.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            patches[row['dataset_id']]+=1
            assert int(row['x'])>=0 and int(row['y'])>=0 and row['width']=='480' and row['height']=='96'
    for d in datasets:
        checks.append(dict(check='normalized_sample_and_patch_counts',id=d['id'],passed=actual[d['id']]==d['samples'] and patches[d['id']]==d['patches']))
    known={d['id'] for d in datasets}
    checks.append(dict(check='model_bindings_resolve',passed=all(set(m['dataset_ids'])<=known for m in models)))
    if (OUT/'raw_roots.json').is_file():
        raw_counts=Counter()
        with gzip.open(OUT/'raw_files.csv.gz','rt',encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):raw_counts[row['root_id']]+=1
        for row in json.loads((OUT/'raw_roots.json').read_text()):
            checks.append(dict(check='raw_directory_inventory_count',id=row['id'],passed=raw_counts[row['id']]==row['files'] and not row['errors']))
    report=dict(checks=len(checks),passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],details=checks)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(checks=report['checks'],passed=report['passed'],failed=report['failed'],models_without_weight_lookup=[m['name'] for m in models if not m['run_artifacts_present']],metadata_snapshots=len(evidence))),flush=True)
    if report['failed']:raise SystemExit(1)


if __name__=='__main__':main()
