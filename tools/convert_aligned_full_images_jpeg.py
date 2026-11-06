"""Create a geometry-preserving quality95 JPEG derivative of the verified full-image ZIP."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

from PIL import Image,JpegImagePlugin

SOURCE_ARCHIVE=None


def sha256(path):
    """@luna-comment
    Input: existing file. Output: SHA256 of exact stored bytes.
    Key Steps: sequential8MiB reads with bounded memory.
    Key Params: used for source and final ZIP identity. State/Dependencies: read-only file.
    """
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save_json(path,value):
    """@luna-comment
    Input: own receipt path and JSON data. Output: atomically published UTF8 state.
    Key Steps: write temporary sibling and replace own destination.
    Key Params: partial progress is not visible. State/Dependencies: output root only.
    """
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def init_worker(source):
    """@luna-comment
    Input: immutable original ZIP path. Output: process-owned read handle.
    Key Steps: open independent ZIP per worker, avoiding shared seek offsets.
    Key Params: one codec operation per worker; no GPU.
    State/Dependencies: SOURCE_ARCHIVE owned until worker exit; source never modified.
    """
    global SOURCE_ARCHIVE
    SOURCE_ARCHIVE=zipfile.ZipFile(source)


def convert_one(job):
    """@luna-comment
    Input: ZIP member name and authoritative original PNG SHA256.
    Output: JPEG bytes, original size, verified source hash and updated content hash.
    Key Steps: read/CRC/hash PNG; decode RGB; encode JPEG; independently inspect header.
    Key Params: quality95,subsampling0(4:4:4),baseline/nonprogressive,unchanged resolution.
    State/Dependencies: process-owned ZIP/Pillow only; no enhancement or resizing.
    """
    name,expected=job;raw=SOURCE_ARCHIVE.read(name)
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('PNG source hash mismatch: '+name)
    with Image.open(io.BytesIO(raw)) as original:
        size=original.size;rgb=original.convert('RGB')
    buffer=io.BytesIO();rgb.save(buffer,format='JPEG',quality=95,subsampling=0,optimize=False,progressive=False)
    encoded=buffer.getvalue()
    with Image.open(io.BytesIO(encoded)) as check:
        if check.format!='JPEG' or check.size!=size or JpegImagePlugin.get_sampling(check)!=0:
            raise ValueError('JPEG encoding/geometry mismatch')
        check.load()
    return name,encoded,dict(width=size[0],height=size[1],source_png_sha256=expected,sha256=hashlib.sha256(encoded).hexdigest(),bytes=len(encoded))


def convert_zip(source,output,workers=8):
    """@luna-comment
    Input: verified original ZIP and isolated new output root; bounded CPU worker count.
    Output: verified JPEG ZIP with updated portable manifests/hashes and PNG lineage.
    Key Steps: verify source archive; stream at most2xworkers JPEG jobs; one ZIP writer;
    copy unchanged OCR/masks; regenerate metadata; verify CRC and final archive SHA256.
    Key Params: LQ/GT only become JPG; auxiliary masks stay lossless PNG; original retained.
    State/Dependencies: refuses existing output/partial; partial is unpublished on failure.
    """
    output.mkdir(parents=True,exist_ok=True)
    final=output/'aligned_training_full_images_jpg_q95_20261021.zip';partial=final.with_suffix('.zip.partial')
    if final.exists() or partial.exists():raise FileExistsError('JPEG destination exists')
    started=time.monotonic();prefix='aligned_training_full_images_20261021/'
    source_hash=sha256(source);save_json(output/'status.json',dict(phase='converting',images=0))
    with zipfile.ZipFile(source) as original:
        rows={split:[json.loads(line) for line in original.read(prefix+'manifests/'+split+'.jsonl').decode().splitlines()] for split in ('train','val')}
        inventory=[json.loads(line) for line in original.read(prefix+'asset_inventory.jsonl').decode().splitlines()]
        jobs=[(prefix+entry['path'],entry['sha256']) for entry in inventory if entry['kind'] in ('lq','gt')]
        converted={};updated=[];hashes=[];output_image_bytes=0
        with zipfile.ZipFile(partial,'x',allowZip64=True) as target,ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(source),)) as pool:
            pending={pool.submit(convert_one,job) for job in list(jobs[:min(len(jobs),workers*2)])}
            iterator=iter(jobs[min(len(jobs),workers*2):])
            while pending:
                ready,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    name,encoded,info=future.result();relative=name[len(prefix):];new=relative[:-4]+'.jpg'
                    target.writestr(prefix+new,encoded,compress_type=zipfile.ZIP_STORED)
                    converted[relative]=dict(info,path=new);output_image_bytes+=len(encoded)
                    next_job=next(iterator,None)
                    if next_job is not None:pending.add(pool.submit(convert_one,next_job))
                if len(converted)%50<len(ready) or not pending:
                    state=dict(phase='converting',images=len(converted),expected_images=len(jobs),jpeg_bytes=output_image_bytes,seconds=time.monotonic()-started)
                    save_json(output/'status.json',state);print(json.dumps(state),flush=True)
            for entry in inventory:
                entry=dict(entry);relative=entry['path']
                if relative in converted:
                    info=converted[relative];entry.update(info);entry['encoding']='JPEG_Q95_444'
                else:
                    data=original.read(prefix+relative)
                    if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Auxiliary source hash mismatch')
                    target.writestr(prefix+relative,data,compress_type=zipfile.ZIP_DEFLATED if relative.endswith('.json') else zipfile.ZIP_STORED)
                updated.append(entry);hashes.append(entry['sha256']+'  '+entry['path'])
            for split,items in rows.items():
                for row in items:
                    row['source_png_files']={kind:row['files'][kind] for kind in ('lq','gt')}
                    row['source_output_sha256']=dict(row['output_sha256'])
                    row['image_encoding']=dict(format='JPEG',quality=95,chroma_subsampling='4:4:4',lossy=True,resized=False)
                    for kind in ('lq','gt'):
                        info=converted[row['files'][kind]]
                        if [info['width'],info['height']]!=row['cropped_size_wh']:raise ValueError('Source manifest dimensions mismatch')
                        row['files'][kind]=info['path'];row['output_sha256'][kind]=info['sha256']
                target.writestr(prefix+'manifests/'+split+'.jsonl',''.join(json.dumps(row)+'\n' for row in items),compress_type=zipfile.ZIP_DEFLATED)
            target.writestr(prefix+'checksums.sha256','\n'.join(hashes)+'\n',compress_type=zipfile.ZIP_DEFLATED)
            target.writestr(prefix+'source_png_checksums.sha256',original.read(prefix+'checksums.sha256'),compress_type=zipfile.ZIP_DEFLATED)
            target.writestr(prefix+'asset_inventory.jsonl',''.join(json.dumps(row)+'\n' for row in updated),compress_type=zipfile.ZIP_DEFLATED)
            provenance=json.loads(original.read(prefix+'provenance.json'));provenance['jpeg_derivative']=dict(source_zip_sha256=source_hash,quality=95,chroma_subsampling='4:4:4',images=len(jobs),resized=False)
            target.writestr(prefix+'provenance.json',json.dumps(provenance,indent=2)+'\n',compress_type=zipfile.ZIP_DEFLATED)
            target.writestr(prefix+'README.md','# JPEG quality95 aligned full-image derivative\n\nLQ/GT:JPEG quality95,4:4:4,original resolution. No preprocessing,augmentation or text-line cropping. JPEG is lossy; the original PNG ZIP remains unchanged. OCR/geometry/masks/splits preserved; masks remain lossless PNG. Updated manifests/checksums identify JPEG files and retain original PNG hashes as lineage.\n',compress_type=zipfile.ZIP_DEFLATED)
    save_json(output/'status.json',dict(phase='crc_verification',images=len(jobs)))
    with zipfile.ZipFile(partial) as archive:
        bad=archive.testzip()
        if bad is not None:raise ValueError('Output CRC failure: '+bad)
        names=archive.namelist()
        if len(names)!=len(set(name.casefold() for name in names)):raise ValueError('Windows filename collision')
        members=len(names)
    save_json(output/'status.json',dict(phase='archive_hash',images=len(jobs)))
    result=dict(complete=True,archive=str(final),archive_bytes=partial.stat().st_size,archive_sha256=sha256(partial),
                source_zip_sha256=source_hash,source_zip_bytes=source.stat().st_size,images=len(jobs),quality=95,chroma_subsampling='4:4:4',
                members=members,train_pairs=len(rows['train']),val_pairs=len(rows['val']),dimensions_verified=True,crc_verified=True,
                seconds=time.monotonic()-started)
    partial.replace(final);save_json(output/'COMPLETE.json',result);save_json(output/'status.json',dict(phase='complete',**result))
    print(json.dumps(result),flush=True)
    return result


def main():
    """@luna-comment
    Input: source ZIP,new output directory,CPU worker count. Output: verified JPEG ZIP.
    Key Steps: parse explicit paths; reject invalid worker counts; perform one conversion.
    Key Params:8workers default; no GPU or training dependencies.
    State/Dependencies: standard library/Pillow; local transfer is a separate verified step.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    convert_zip(args.source,args.output,args.workers)


if __name__=='__main__':main()
