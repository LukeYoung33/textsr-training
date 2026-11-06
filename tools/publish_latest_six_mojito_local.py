"""Local six-panel view over audited NAS assets without copying full-resolution images."""
from pathlib import Path
import sys,json,argparse,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=Path('Y:/guang.yang/TextSR/data/latest_six_500k_mojito280_20261103')
OUT=ROOT/'evaluation/latest_six_500k_mojito280_20261103'


def main(preview=False):
    """@luna-comment Input:audited final manifest or one-image acceptance pilot. Output:local6panel viewer. Key Steps:require COMPLETE/AUDIT forfull;copy metadata;adapt rendered panelcount/CSS only. Key Params:all6 post variants default,14choices. State/Dependencies:read-only Y images,existing viewer renderer unchanged."""
    from tools.build_camera_2x2_viewer import render_html
    name='PILOT_MANIFEST.json' if preview else 'manifest.json';data=json.loads((SOURCE/name).read_text(encoding='utf-8'))
    if not preview:
        assert json.loads((SOURCE/'COMPLETE.json').read_text(encoding='utf-8'))['passed']
        assert json.loads((SOURCE/'AUDIT.json').read_text(encoding='utf-8'))['passed'] and len(data['rows'])==280
        for filename in ['manifest.json','AUDIT.json','COMPLETE.json','MODEL_BINDINGS.json','RUN_BINDING.json','selection.json']:
            shutil.copyfile(SOURCE/filename,OUT/filename)
    data['defaults']='v3m_guard,v3m_l1,v3m_mix50,v5_guard,v5_l1,v5_mix50'
    text=render_html(data)
    assert text.count('for(let i=0;i<4;i++)')==1 and text.count('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;')==1
    text=text.replace('for(let i=0;i<4;i++)','for(let i=0;i<6;i++)').replace('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    (OUT/('preview.html' if preview else 'index.html')).write_text(text,encoding='utf-8')
    (OUT/('PILOT_MANIFEST.json' if preview else 'manifest.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args();main(args.preview)
