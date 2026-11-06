"""Read-only live document/OCR/restoration pipeline status page."""
import html
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_offline_camera_matrix import ARMS,save_json
OUT=ROOT.parent


def main():
    """@luna-comment
    Input: this task's stage/progress records. Output:30s HTML/JSON updates until terminal.
    Key Steps: count completed doc/OCR/prep artifacts, sum disjoint inference progress,
    expose failure or gallery links without changing any running process.
    Key Params:280 source frames; all retained instances counted separately.
    State/Dependencies: read-only stage tree except own progress page/receipt.
    """
    while True:
        path=OUT/'pipeline_status.json'
        pipeline=json.loads(path.read_text()) if path.exists() else dict(phase='starting')
        documents=len(list((OUT/'documents').glob('*/result.json')))
        ocr=len(list((OUT/'ocr').glob('*/*.json')))
        prepared=len(list((OUT/'prepared').glob('*/*.receipt.json')))
        progress={}
        for arm in ARMS:
            progress[arm]=sum(json.loads(p.read_text())['done'] for p in (OUT/arm).glob('progress.rank*.json'))
        state=dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),pipeline=pipeline,document_frames=documents,
                   warped_ocr_instances=ocr,prepared_instances=prepared,restoration_frames=progress)
        save_json(OUT/'LIVE_PROGRESS.json',state)
        rows=''.join(f'<tr><td>{arm}</td><td>{done}/280</td></tr>' for arm,done in progress.items())
        body='<h1>Document warp/unwarp · Camera evaluation</h1><p>Phase: '+html.escape(pipeline['phase'])+'</p><p>Document records: '+str(documents)+'/280 · Warped OCR instances: '+str(ocr)+' · Prepared instances: '+str(prepared)+'</p><table><tr><th>Restoration arm</th><th>Frames</th></tr>'+rows+'</table><p>All original/model/document/OCR/pre/post/unwarp stages are retained. Same four final250K checkpoints as the previous baseline.</p><p><a href="index.html">Comparison gallery (available after rendering)</a></p><button onclick="location.reload()">Refresh now</button><p>Automatic refresh every30 seconds · Updated '+state['updated_at']+'</p>'
        if pipeline['phase']=='failed':
            body+='<pre>'+html.escape(json.dumps(pipeline,indent=2))+'</pre>'
        page='<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Document camera pipeline</title><style>body{font:16px/1.7 Segoe UI;margin:40px;color:#183247;background:#f3f6fa}table{background:white;border-collapse:collapse}td,th{padding:12px 24px;border-bottom:1px solid #ddd}button{padding:10px 20px}pre{white-space:pre-wrap}</style>'+body+'</html>'
        (OUT/'status.html').write_text(page,encoding='utf-8')
        if pipeline['phase'] in ('complete','failed'):
            return
        time.sleep(30)


if __name__=='__main__':
    main()
