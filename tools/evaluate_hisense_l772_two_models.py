"""Reuse completed two-model pipeline for the explicit68 HisenseL772 JPGs."""
from pathlib import Path
import sys,json,shutil,inspect,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import evaluate_zoom_plate_two_models as core
from tools.run_uniform20_steep_scurve import save,digest
PREVIOUS=core.OUT
core.OUT=core.DATA/'hisense_l772_two_models_20261105';core.__file__=__file__


def prepare():
    """@luna-comment Input:localhash inventory andtransferred68JPEGs. Output:frozen source/model/code bindings. Key Steps:verifylocal-remote bytes/dimensions/orientation;reuse prior exact500Kmodels. Key Params:68x6stages,storednormalorientation. State/Dependencies:original input bytes untouched."""
    from PIL import Image
    out=core.OUT;inventory=json.loads((out/'source_inventory.json').read_text());assert len(inventory['samples'])==68
    files={p.name:p for p in (out/'source_upload').rglob('*.jpg')};assert set(files)=={r['filename'] for r in inventory['samples']}
    samples=[]
    for r in inventory['samples']:
        p=files[r['filename']];assert digest(p)==r['sha256'] and p.stat().st_size==r['bytes']
        with Image.open(p) as im:assert im.size==(r['width'],r['height']) and im.getexif().get(274,1)==r['orientation']==1
        key='hisense_l772/'+p.stem
        samples.append(dict(id=key,group='hisense_l772',stem=p.stem,source=str(p),original_local_source=r['local_source'],source_sha256=r['sha256'],format='JPG',width=r['width'],height=r['height'],stride=None,orientation=1,output=str(out/'output'/(key+'.png')),input=str(out/'preprocess'/(key+'.png'))))
    assert json.loads((PREVIOUS/'COMPLETE.json').read_text())['passed'];bindings=json.loads((PREVIOUS/'MODEL_BINDINGS.json').read_text());(out/'models').mkdir(exist_ok=True)
    for arm,b in bindings.items():
        for key,suffix in [('config','.yml'),('checkpoint','.pth')]:
            source=Path(b[key]);assert digest(source)==b[key+'_sha256'];target=out/'models'/(arm+suffix);shutil.copy2(source,target);assert digest(target)==b[key+'_sha256'];b[key]=str(target)
    save(out/'MODEL_BINDINGS.json',bindings);save(out/'selection.json',dict(samples=samples))
    run=json.loads((PREVIOUS/'RUN_BINDING.json').read_text());run.update(selection_sha256=digest(out/'selection.json'),groups={'hisense_l772':68},outputs=408,nv12_conversion='not applicable;68 JPEG sources,allorientation1',local_source_inventory_sha256=digest(out/'source_inventory.json'))
    for p,sha in run['code_sha256'].items():assert digest(ROOT/p)==sha
    save(out/'RUN_BINDING.json',run)


def bind():
    """@luna-comment Input:accepted28image controller. Output:isolated68image counters/audit. Key Steps:checked literal-only population replacements;preserve actual inference/pre/post functions. Key Params:28->68,168->408. State/Dependencies:process-local module only."""
    progress=inspect.getsource(core.progress);assert '168' in progress and '/28' in progress
    progress=progress.replace('168','408').replace('/28','/68').replace('Zoom/车牌','海信L772')
    audit=inspect.getsource(core.audit);assert 'assert len(outputs)==28' in audit and 'samples=28,outputs=168' in audit
    audit=audit.replace('assert len(outputs)==28','assert len(outputs)==68').replace('samples=28,outputs=168','samples=68,outputs=408')
    namespace=dict(core.__dict__);exec(compile(progress,'<hisense_progress>','exec'),namespace);exec(compile(audit,'<hisense_audit>','exec'),namespace)
    core.progress=namespace['progress'];core.audit=namespace['audit'];core.prepare=prepare


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','preprocess','worker']);p.add_argument('--arm',choices=list(core.LABELS));p.add_argument('--rank',type=int,default=0);args=p.parse_args();bind()
    if args.mode=='worker':core.worker(args.arm,args.rank)
    elif args.mode=='preprocess':core.preprocess()
    else:core.run()
