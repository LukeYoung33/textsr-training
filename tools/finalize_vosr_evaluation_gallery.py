"""Add explicit intermediate viewers and table downloads to the completed gallery."""
import html
import json
import os
from pathlib import Path

OUT = Path('/AI_DNS/guang.yang/TextSR/data/vosr_final_evaluation_20261022')


def image_link(path, page):
    """@luna-comment Input:asset/page paths. Output:relative browser URL. Key Steps:filesystem-relative normalization. Key Params:same mapped share. State/Dependencies:no credentials or HTTP server."""
    return html.escape(os.path.relpath(path,page.parent).replace(os.sep,'/'),quote=True)


def main():
    """@luna-comment
    Input:complete gallery/receipts. Output:per-result stage viewers with native PNG/reference links.
    Key Steps:link immutable caches and own stages; lazy-load native images; check each stage asset exists.
    Key Params:863x6 viewers, no source image copies. State/Dependencies:completed independent audit.
    """
    audit = json.loads((OUT/'INDEPENDENT_AUDIT.json').read_text())
    if audit['status']!='passed':
        raise ValueError('Independent audit not passed')
    count = 0
    for suite in ('camera','validation'):
        for receipt_file in sorted((OUT/suite).glob('*/receipts/*.json')):
            receipt = json.loads(receipt_file.read_text())
            arm,identity = receipt['arm'],receipt['id']
            page = OUT/suite/arm/'stages'/identity/'index.html'
            assets = [('Final original-size result',OUT/receipt['final'])]
            if suite=='validation':
                assets += [('Original LQ',Path(receipt['source_files']['lq'])),('Raw GT_VOSR',Path(receipt['source_files']['gt'])),('Model input',page.parent/'input.png'),('Model output',page.parent/'model.png')]
            else:
                record_path = Path(receipt['document_record'])
                cache = record_path.parents[2]
                assets += [('Original camera',Path(receipt['source_file'])),('Document detection',record_path.parent/'overlay.png'),('Owned support',page.parent/'ownership.png')]
                for stage in receipt['stages']:
                    iid = f"{stage['instance_id']:03d}"
                    ocr = json.loads((cache/'ocr'/identity/(iid+'.json')).read_text())
                    assets += [('Document '+iid+' warped original',Path(ocr['warp_file']))]
                    assets += [('Document '+iid+' '+label,page.parent/iid/(label+'.png')) for label in ('input','model','post','alpha')]
            cards = []
            for title,path in assets:
                if not path.is_file():
                    raise ValueError('Missing stage asset '+str(path))
                link = image_link(path,page)
                cards.append('<section><h3>'+html.escape(title)+'</h3><a href="'+link+'">Native image</a><img loading="lazy" src="'+link+'"></section>')
            content = '<!doctype html><meta charset="utf-8"><title>Intermediate stages</title><style>body{font:16px sans-serif}main{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}img{width:100%}pre{white-space:pre-wrap}</style><h1>'+html.escape(suite+' / '+arm+' / '+identity)+'</h1><main>'+''.join(cards)+'</main><h2>Provenance and strategy</h2><pre>'+html.escape(json.dumps(receipt,indent=2))+'</pre>'
            page.write_text(content,encoding='utf-8')
            comparison = OUT/'pages'/(suite+'_'+identity+'.html')
            old = '../'+suite+'/'+arm+'/stages/'+identity+'/'
            comparison.write_text(comparison.read_text().replace('href="'+old+'"','href="'+old+'index.html"'),encoding='utf-8')
            count += 1
    index = OUT/'index.html'
    index.write_text(index.read_text().replace('<h1>VOSR six final models</h1>','<h1>VOSR six final models</h1><p><a href="validation.html">Validation performance report</a> | <a href="validation_per_image.csv">Per-image CSV</a> | <a href="VALIDATION_SUMMARY.json">Global and per-dataset metrics</a> | <a href="INDEPENDENT_AUDIT.json">Independent audit</a> | <a href="STATUS.html">Progress</a></p>'),encoding='utf-8')
    (OUT/'GALLERY_AUDIT.json').write_text(json.dumps(dict(status='passed',intermediate_viewers=count,comparison_pages=863,asset_links_checked=True)))
    print('Gallery stage viewers:',count)


if __name__=='__main__':
    main()
