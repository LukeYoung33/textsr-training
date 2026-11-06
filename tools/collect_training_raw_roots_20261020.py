"""Enumerate bounded original-photo roots without copying image data."""
import csv
import gzip
import json
import os
from pathlib import Path
import time

from build_training_dataset_handoff_20261020 import OUT, DATA, windows


def main():
    """@luna-comment Input:explicit historical raw photo/reference directories.
    Output:complete raw file list and root-level counts/existence/size metadata.
    Key Steps:walk named roots without following directory symlinks; retain Linux-only
    locations; record enumeration errors. Key Params:no image decoding or copying.
    State/Dependencies:read-only source trees and isolated handoff metadata output.
    """
    roots = [
        ('0921', 'raw_LQ', DATA/'Lenovo_data_260912/LQ'),
        ('0921', 'raw_GT', DATA/'Lenovo_data_260912/GT'),
        ('0928/1014/1015', 'raw_LQ', DATA/'Lenovo_data_260926/source/LQ'),
        ('0928/1014/1015', 'raw_GT', DATA/'Lenovo_data_260926/source/GT'),
        ('1014/1015', 'raw_camera', DATA/'lenovo_data_261011/Camera'),
        ('1014/1015', 'raw_camera_A', DATA/'HX_data_1013/PhotosA'),
        ('1014/1015', 'raw_camera_B', DATA/'HX_data_1013/PhotosB'),
        ('1014/1015', 'rendered_display_reference', Path('/AI_DNS/guang.yang/lenovo_20261010_500groups/display_images')),
        ('1014/1015', 'original_reference_and_annotations', Path('/data/guang.yang/textsr/lenovo_261011_alignment/source_gt')),
    ]
    OUT.mkdir(parents=True, exist_ok=True);summary=[]
    with gzip.open(OUT/'raw_files.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['root_id','family','role','linux_path','windows_path','relative_path','bytes','mtime_unix'])
        for number,(family,role,root) in enumerate(roots,1):
            row=dict(id=f'R{number:02d}', family=family,role=role,linux_path=str(root),windows_path=windows(root),exists=root.is_dir(),files=0,image_files=0,bytes=0,errors=[])
            def walk_error(error):
                """@luna-comment Input:walk error. Output:explicit error evidence.
                Key Steps:append error. Key Params:no silent exclusions. State:own row.
                """
                row['errors'].append(str(error))
            if row['exists']:
                for directory,_,files in os.walk(root,onerror=walk_error,followlinks=False):
                    for name in sorted(files):
                        path=Path(directory)/name
                        try:stat=path.stat()
                        except OSError as error:row['errors'].append(str(error));continue
                        writer.writerow([row['id'],family,role,str(path),windows(path),str(path.relative_to(root)),stat.st_size,int(stat.st_mtime)])
                        row['files']+=1;row['bytes']+=stat.st_size
                        row['image_files']+=path.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
            summary.append(row);print(row['id'],role,row['files'],row['bytes'],flush=True)
    (OUT/'raw_roots.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('DONE',sum(row['files'] for row in summary),flush=True)


if __name__=='__main__':main()
