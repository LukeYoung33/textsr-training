"""Export unchanged aligned full-image parents of the current training bank to ZIP64."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import zipfile


def sha256(path):
    """@luna-comment
    Input: existing readable file path. Output: lowercase SHA256 of exact file bytes.
    Key Steps: sequential8MiB blocks; no decoding or mutation.
    Key Params: bounded memory regardless of ZIP size. State/Dependencies: standard library.
    """
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def select_parents(source,bank):
    """@luna-comment
    Input: original aligned full-image root and actual offline-bank root.
    Output: exact nonempty train/val records plus source/bank hashes and exclusions.
    Key Steps: intersect manifest IDs with positive-count bank shards; enforce source
    stage, split, unique membership and required LQ/GT/OCR/mask identities.
    Key Params: no augmentation/preprocessing; display_yuv_fit_v1_20261014 only.
    State/Dependencies: read-only authoritative JSON files; no source transformations.
    """
    selected={};provenance={};identities={}
    for split in ('train','val'):
        bank_path=bank/(split+'.json');manifest_path=source/'manifests'/(split+'.jsonl')
        config=json.loads(bank_path.read_text())
        ids={row['sample_id'] for row in config['shards'] if row['count']>0}
        rows=[json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        included=[row for row in rows if row['sample_id'] in ids]
        if len(included)!=len(ids) or len({row['sample_id'] for row in included})!=len(ids):
            raise ValueError('Parent membership mismatch: '+split)
        for row in included:
            if row['bucket']!=split or row['preprocess']['applied'] is not False or row['photometry_version']!='display_yuv_fit_v1_20261014':
                raise ValueError('Wrong source stage or split')
            if set(row['files'])!={'lq','gt','ocr','text_mask','valid_mask'}:
                raise ValueError('Unexpected source file schema')
        selected[split]=included
        identities[split]={row['source_identity'] for row in included}
        provenance[split]=dict(parents=len(included),source_manifest_sha256=sha256(manifest_path),bank_manifest_sha256=sha256(bank_path),
                               excluded_empty_ids=[row['sample_id'] for row in rows if row['sample_id'] not in ids])
    if identities['train']&identities['val']:
        raise ValueError('Source identity train/val overlap')
    return selected,provenance


def save_json(path,value):
    """@luna-comment
    Input: own output JSON path and serializable state. Output: atomic UTF8 receipt.
    Key Steps: write sibling temporary then replace only own destination.
    Key Params: no partially visible progress/completion JSON.
    State/Dependencies: export directory only; source files remain read-only.
    """
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def export_zip(source,bank,output):
    """@luna-comment
    Input: authoritative aligned source/bank roots and isolated export directory.
    Output: verified ZIP64, per-file hashes/manifests and archive SHA256 receipt.
    Key Steps: select parents; stream unchanged files while validating hashes/PNG
    full-image dimensions; include portable records; verify every CRC; hash and publish.
    Key Params: original PNGs stored without re-encoding; JSON compressed;8MiB buffers.
    State/Dependencies: refuses existing archive/partial; leaves failed partial unpublished;
    no training, source or model changes. Existing alignment common ROI is retained.
    """
    output.mkdir(parents=True,exist_ok=True)
    final=output/'aligned_training_full_images_20261021.zip'
    partial=final.with_suffix('.zip.partial')
    if final.exists() or partial.exists():
        raise FileExistsError('Export destination already contains an archive or partial')
    rows,provenance=select_parents(source,bank)
    prefix='aligned_training_full_images_20261021'
    hashes=[];entries=[];files=0;bytes_read=0;started=time.monotonic()
    expected=sum(len(items)*5 for items in rows.values())
    save_json(output/'status.json',dict(phase='packing',files=0,expected_files=expected,source_bytes=0))
    with zipfile.ZipFile(partial,'x',allowZip64=True) as archive:
        for split,items in rows.items():
            for row in items:
                for kind,relative in row['files'].items():
                    path=source/relative
                    if Path(relative).is_absolute() or not path.resolve().is_relative_to(source.resolve()):
                        raise ValueError('Source path escapes aligned image root')
                    name=prefix+'/'+relative
                    info=zipfile.ZipInfo.from_file(path,arcname=name)
                    info.compress_type=zipfile.ZIP_DEFLATED if path.suffix=='.json' else zipfile.ZIP_STORED
                    digest=hashlib.sha256();size=0
                    with path.open('rb') as stream,archive.open(info,'w',force_zip64=True) as target:
                        first=True
                        for block in iter(lambda:stream.read(8*1024*1024),b''):
                            if first and kind!='ocr':
                                if block[:8]!=b'\x89PNG\r\n\x1a\n' or tuple(struct.unpack('>II',block[16:24]))!=tuple(row['cropped_size_wh']):
                                    raise ValueError('Not authoritative full-image PNG dimensions: '+relative)
                            first=False;digest.update(block);target.write(block);size+=len(block)
                    actual=digest.hexdigest()
                    if actual!=row['output_sha256'][kind]:
                        raise ValueError('Source asset SHA256 mismatch: '+relative)
                    hashes.append(actual+'  '+relative)
                    entries.append(dict(path=relative,bytes=size,sha256=actual,kind=kind,split=split,sample_id=row['sample_id']))
                    files+=1;bytes_read+=size
                if len(entries)%250==0:
                    state=dict(phase='packing',files=files,expected_files=expected,source_bytes=bytes_read,seconds=time.monotonic()-started)
                    save_json(output/'status.json',state);print(json.dumps(state),flush=True)
            content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in items)
            archive.writestr(prefix+'/manifests/'+split+'.jsonl',content.encode('utf-8'),compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
        archive.writestr(prefix+'/checksums.sha256','\n'.join(hashes)+'\n',compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
        archive.writestr(prefix+'/asset_inventory.jsonl',''.join(json.dumps(entry)+'\n' for entry in entries),compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
        archive.writestr(prefix+'/provenance.json',json.dumps(provenance,indent=2)+'\n',compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
        readme=f'''# Aligned full-image training parents\n\nTrain:{len(rows['train'])} pairs; validation:{len(rows['val'])} pairs.\n\nThese are original published spatially and photometrically aligned LQ/GT full-image pairs used as parents of this training round, before adaptive preprocessing, augmentation and text-line/patch cropping. Original PNG/JSON bytes are preserved. Spatial alignment may already have selected a common valid full-image ROI (recorded as crop_xywh); these are not unregistered sensor originals.\n\nDirectories: train/ and val/, each with LQ,GT,OCR,text_mask,valid_mask. Relative files in manifests resolve from this archive root. output_sha256 and checksums.sha256 verify every original asset. Source provenance absolute paths are historical audit references, not required for opening this export.\n\nMembership comes from positive-count bank shards. Two empty train parents were not used: capture_261011_004394 and capture_261011_004871. No review-excluded data are included.\n'''
        archive.writestr(prefix+'/README.md',readme,compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
    save_json(output/'status.json',dict(phase='crc_verification',files=files,source_bytes=bytes_read))
    with zipfile.ZipFile(partial) as archive:
        bad=archive.testzip()
        if bad is not None:
            raise ValueError('ZIP CRC failure: '+bad)
        members=len(archive.infolist())
    save_json(output/'status.json',dict(phase='archive_hash',files=files,source_bytes=bytes_read))
    archive_hash=sha256(partial)
    partial.replace(final)
    receipt=dict(complete=True,archive=str(final),archive_bytes=final.stat().st_size,archive_sha256=archive_hash,
                 files=files,members=members,source_bytes=bytes_read,splits=provenance,crc_verified=True,
                 all_asset_hashes_verified=True,all_full_image_dimensions_verified=True,seconds=time.monotonic()-started)
    save_json(output/'COMPLETE.json',receipt);save_json(output/'status.json',dict(phase='complete',**receipt))
    print(json.dumps(receipt),flush=True)
    return receipt


def main():
    """@luna-comment
    Input: explicit source,bank and new output CLI paths. Output: verified full-image ZIP.
    Key Steps: parse required paths and execute source-preserving export once.
    Key Params: no local destination/transport implicit in exporter; downloads separate.
    State/Dependencies: Python standard library only, no GPU/model imports.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export_zip(args.source,args.bank,args.output)


if __name__=='__main__':
    main()
