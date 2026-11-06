"""Independently audit final six-arm receipts and export common-domain validation tables."""
import csv
import html
import json
import math
from pathlib import Path

OUT = Path('/AI_DNS/guang.yang/TextSR/data/vosr_final_evaluation_20261022')
ARMS = ('01_v5_preprocessed','02_v3_preprocessed','03_v5_raw','04_v3_raw','05_v5_block256_raw','06_v3_block256_raw')


def aggregate(records):
    """@luna-comment
    Input:common-support image metric records. Output:pooled and macro statistics.
    Key Steps:sum raw SSE/count before PSNR; separately average finite image PSNR/SSIM.
    Key Params:255 RGB range; no infinity substitution. State/Dependencies:receipt-only audit.
    """
    result = dict(images=len(records))
    for region in ('all','text','background'):
        sse = sum(row[region]['sse'] for row in records)
        count = sum(row[region]['count'] for row in records)
        values = [row[region]['psnr'] for row in records if row[region]['psnr'] is not None]
        result[region] = dict(pooled_psnr=10*math.log10(255**2/(sse/count)) if sse and count else None,
                              macro_finite_psnr=sum(values)/len(values) if values else None,
                              finite_images=len(values),channel_samples=count,sse=sse)
    count = sum(row['ssim_y']['count'] for row in records)
    values = [row['ssim_y']['sum']/row['ssim_y']['count'] for row in records if row['ssim_y']['count']]
    result['ssim_y'] = dict(pooled=sum(row['ssim_y']['sum'] for row in records)/count if count else None,
                            macro=sum(values)/len(values) if values else None,windows=count)
    return result


def main():
    """@luna-comment
    Input:complete six-arm camera/validation receipts. Output:audited per-image CSV and dataset aggregates.
    Key Steps:require equal identities/source hashes/baselines, check native canvas/tonal routing, aggregate.
    Key Params:280/583 per arm. State/Dependencies:no model/source edits; GT never enters inference.
    """
    summary = {}
    reference = {}
    camera_reference = {}
    csv_rows = []
    bindings = {}
    for arm in ARMS:
        records = [json.loads(path.read_text()) for path in sorted((OUT/'validation'/arm/'receipts').glob('*.json'))]
        camera = [json.loads(path.read_text()) for path in sorted((OUT/'camera'/arm/'receipts').glob('*.json'))]
        if len(records)!=583 or len(camera)!=280:
            raise ValueError('Incomplete arm '+arm)
        manifest = OUT.parent/'adaptive_vosr_preprocessed_augmented_v1_20261021/source_view/manifests/val.jsonl'
        expected = {json.loads(line)['sample_id'] for line in manifest.read_text().splitlines()}
        if {row['id'] for row in records}!=expected or {row['id'] for row in camera}!={f'{i:03d}' for i in range(1,281)}:
            raise ValueError('Evaluation identity coverage mismatch')
        binding = records[0]['binding']
        bindings[arm] = binding
        groups = {'all':[]}
        for row in records:
            identity = row['id']
            evidence = dict(source_hashes=row['source_hashes'],baseline=row['baseline'],source_identity=row['source_identity'],dataset=row['dataset'])
            if arm==ARMS[0]:
                reference[identity] = evidence
            elif evidence!=reference.get(identity):
                raise ValueError('Cross-arm source/mask/baseline mismatch '+identity)
            if row['binding']!=binding:
                raise ValueError('Mixed model provenance')
            strategy = row['stages'][0]['restoration']
            if strategy['canvas_hw']!=([256,256] if 'block256' in arm else [96,480]):
                raise ValueError('Wrong inference canvas')
            if (strategy['preprocessing']=='none') != ('preprocessed' not in arm):
                raise ValueError('Wrong tonal strategy')
            for label,value in [('model',row['metrics'])]+([('LQ_baseline',row['baseline'])] if arm==ARMS[0] else []):
                csv_rows.append(dict(arm=arm if label=='model' else label,id=identity,dataset=row['dataset'],source_identity=row['source_identity'],
                    psnr_rgb=value['all']['psnr'],psnr_text=value['text']['psnr'],psnr_background=value['background']['psnr'],
                    ssim_y=value['ssim_y']['sum']/value['ssim_y']['count'] if value['ssim_y']['count'] else None,
                    valid_channel_samples=value['all']['count'],seconds=row['seconds'] if label=='model' else '',final=row['final'] if label=='model' else row['source_files']['lq']))
            groups['all'].append(row['metrics'])
            groups.setdefault('dataset:'+row['dataset'],[]).append(row['metrics'])
        for row in camera:
            if row['binding']!=binding or not row['outside_owned_unchanged']:
                raise ValueError('Camera provenance/support mismatch')
            evidence = dict(source_sha256=row['source_sha256'],document_record_sha256=row['document_record_sha256'],
                            stages=[(stage['instance_id'],stage['ocr_sha256'],stage['warp_sha256']) for stage in row['stages']])
            if arm==ARMS[0]:
                camera_reference[row['id']] = evidence
                record_path = Path(row['document_record'])
                document = json.loads(record_path.read_text())
                for item in document['instances']:
                    ocr_path = record_path.parents[2]/'ocr'/row['id']/(f"{item['instance_id']:03d}"+'.json')
                    ocr = json.loads(ocr_path.read_text())
                    if not ocr['fresh_warp_detection'] or ocr['transform']!=item['warp']['image_to_output']:
                        raise ValueError('Stale or mismatched warped OCR geometry')
            elif evidence!=camera_reference.get(row['id']):
                raise ValueError('Camera cross-arm source/geometry/OCR mismatch')
        summary[arm] = {key:aggregate(values) for key,values in groups.items()}
    baseline_groups = {'all':[row['baseline'] for row in reference.values()]}
    for row in reference.values():
        baseline_groups.setdefault('dataset:'+row['dataset'],[]).append(row['baseline'])
    summary['LQ_baseline'] = {key:aggregate(values) for key,values in baseline_groups.items()}
    with (OUT/'validation_per_image.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
        writer.writeheader();writer.writerows(csv_rows)
    (OUT/'VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(status='passed',camera=1680,validation=3498,common_source_masks_and_baselines=True,bindings=bindings),indent=2))
    create_html(summary,csv_rows,reference)
    print(json.dumps(summary,indent=2))


def create_html(summary, rows, reference):
    """@luna-comment
    Input:audited aggregates/per-image rows/baselines. Output:readable validation report and paired change tables.
    Key Steps:show global/dataset pooled and macro metrics; link20 largest paired regressions/improvements.
    Key Params:selection by final RGB PSNR delta, not a quality claim. State/Dependencies:complete same583 samples.
    """
    content = '<!doctype html><meta charset="utf-8"><title>VOSR validation performance</title><style>body{font:15px sans-serif;margin:25px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #aaa}th{background:#eee}</style><h1>583-pair validation performance</h1><p><a href="index.html">Six-model image comparison</a> | <a href="validation_per_image.csv">Per-image CSV</a> | <a href="VALIDATION_SUMMARY.json">Full statistics</a></p><p>Final output versus raw GT_VOSR on identical valid support. RGB pixel-pooled PSNR and valid-window Y SSIM; macro scores are separately labeled. Text/background use supplied masks. No OCR recognition accuracy is inferred.</p>'
    for group in summary[ARMS[0]]:
        content += '<h2>'+html.escape(group)+'</h2><table><tr><th>Arm</th><th>Images</th><th>RGB pooled PSNR</th><th>Text pooled PSNR</th><th>Background pooled PSNR</th><th>Y pooled SSIM</th><th>RGB macro PSNR</th></tr>'
        for arm in ('LQ_baseline',)+ARMS:
            stats = summary[arm][group]
            values = [stats[key]['pooled_psnr'] for key in ('all','text','background')]+[stats['ssim_y']['pooled'],stats['all']['macro_finite_psnr']]
            content += '<tr><td>'+arm+'</td><td>'+str(stats['images'])+'</td>'+''.join('<td>'+(f'{value:.5f}' if value is not None else 'exact/empty')+'</td>' for value in values)+'</tr>'
        content += '</table>'
    content += '<h2>Paired per-image changes</h2><p>Largest20 regressions and improvements relative to input LQ, selected by final RGB PSNR difference. These are image agreement diagnostics; use the linked six-model previews and native images for visual assessment.</p>'
    for arm in ARMS:
        values = [(row['psnr_rgb']-reference[row['id']]['baseline']['all']['psnr'],row) for row in rows if row['arm']==arm and row['psnr_rgb'] is not None and reference[row['id']]['baseline']['all']['psnr'] is not None]
        values.sort(key=lambda item:item[0])
        for title,selected in (('Lowest changes',values[:20]),('Highest changes',values[-20:][::-1])):
            content += '<h3>'+arm+' / '+title+'</h3><table><tr><th>Image</th><th>Dataset</th><th>RGB PSNR</th><th>Change dB</th></tr>'
            for delta,row in selected:
                content += '<tr><td><a href="pages/validation_'+html.escape(row['id'],quote=True)+'.html">'+html.escape(row['id'])+'</a></td><td>'+html.escape(row['dataset'])+f'</td><td>{row["psnr_rgb"]:.4f}</td><td>{delta:+.4f}</td></tr>'
            content += '</table>'
    (OUT/'validation.html').write_text(content,encoding='utf-8')


if __name__=='__main__':
    main()
