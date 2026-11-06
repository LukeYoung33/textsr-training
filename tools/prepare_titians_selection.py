"""Materialize frozen-current postprocessing for the Titians selection gallery."""
from pathlib import Path
import sys,json,hashlib,time,os,argparse
from urllib.parse import urlparse,unquote
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=Path(os.environ.get('TITIANS_SELECTION_OUTPUT',str(ROOT/'output/titians_selection_20261030')))
SOURCE=Path(os.environ.get('TITIANS_SELECTION_SOURCE','Y:/guang.yang/TextSR/data/titians_camera_three_20261030'))
KEYS=('v3_input','v5_pre','v5_ft','v3m_ft')


def sha(path):
    """@luna-comment Input:file. Output:SHA256. Key Steps:stream blocks. Key Params:1MiB. State/Dependencies:read-only."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    """@luna-comment Input:owned path/value. Output:atomic JSON. Key Steps:temporary replace. Key Params:UTF8. State/Dependencies:owned artifacts."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def asset(url):
    """@luna-comment Input:trusted manifest URL. Output:mapped file path. Key Steps:decode file scheme. Key Params:no URL network fetch. State/Dependencies:read-only source."""
    u=urlparse(url);assert u.scheme=='file';value=unquote(u.path).lstrip('/')
    if os.name!='nt':value=value.replace('Y:/','/AI_DNS/',1)
    return Path(value)


def process(job):
    """@luna-comment Input:row/model binding. Output:audited native post PNG. Key Steps:hash source;current zero-trial;native and softcap checks;atomic receipt. Key Params:.75/2/24,no USM. State/Dependencies:existing frozen core."""
    import cv2,numpy as np
    from tools.native_y_curve_four_images import process_pair,CurveTrial
    cv2.setNumThreads(1);row,key,core=job;src=asset(row['images'][key]);source_sha=sha(src)
    target=OUT/'post'/key/(row['id']+'.png');receipt=OUT/'receipts'/(key+'_'+row['id']+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text(encoding='utf-8'));assert r['source_sha256']==source_sha and r['core_sha256']==core and sha(target)==r['output_sha256'];return r
    image=cv2.imdecode(np.fromfile(src,np.uint8),cv2.IMREAD_COLOR);assert image.shape[:2]==(row['height'],row['width'])
    result,zero,stats=process_pair(image,CurveTrial(16.,48.,0.,0.));assert np.array_equal(result,zero);assert np.abs(result.astype(np.int16)-image.astype(np.int16)).max()<=24
    target.parent.mkdir(parents=True,exist_ok=True);ok,data=cv2.imencode('.png',result,[cv2.IMWRITE_PNG_COMPRESSION,3]);assert ok
    tmp=target.with_suffix('.tmp');data.tofile(tmp);tmp.replace(target);assert sha(src)==source_sha
    r=dict(id=row['id'],key=key,source=str(src),source_sha256=source_sha,output=str(target),output_sha256=sha(target),core_sha256=core,shape=list(result.shape),stats=stats);save(receipt,r);return r


def main(workers=2):
    """@luna-comment Input:completed160x3 remote results. Output:local post gallery manifest/progress. Key Steps:two workers,resume verified receipts,bind original JPG and four candidates. Key Params:480 outputs. State/Dependencies:immutable NAS source."""
    OUT.mkdir(parents=True,exist_ok=True);m=json.loads((SOURCE/'manifest.json').read_text());assert len(m['rows'])==160
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed'];selection={r['id']:r for r in json.loads((SOURCE/'selection.json').read_text())['samples']}
    core=sha(ROOT/'tools/native_y_curve_four_images.py');started=time.time();jobs=[(r,k,core) for r in m['rows'] for k in KEYS[1:]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n,r in enumerate(pool.map(process,jobs),1):
            save(OUT/'STATUS.json',dict(phase='postprocessing',completed=n,total=480,seconds=time.time()-started,eta_seconds=(time.time()-started)/n*(480-n)));print(n,480,r['id'],r['key'],flush=True)
    for r in m['rows']:
        original=selection[r['id']]['source'];r['original']=original.replace('/AI_DNS/','Y:/',1) if os.name=='nt' else original
        r['lq']=str(asset(r['lq']));r['images']={k:str(asset(r['images'][k])) if k=='v3_input' else str(OUT/'post'/k/(r['id']+'.png')) for k in KEYS}
    m['models']=[x for x in m['models'] if x['key'] in KEYS]
    for model in m['models']:
        if model['key']!='v3_input':model['label']+=' + 固化Y后处理';model['note']='当前部署Y公式 .75 / 2 / 24；原分辨率'
    m.update(title='Titians · 每组选一张',subtitle='V3输入 vs 三模型固化后处理 · 160组',defaults=','.join(KEYS),notes='每组四选一；选择自动保存。全部选完后导出选中PNG和原始输入JPG。三模型统一应用当前部署Y后处理；V3输入不加后处理。')
    save(OUT/'manifest.json',m);save(OUT/'STATUS.json',dict(phase='complete',completed=480,total=480,seconds=time.time()-started,core_sha256=core))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();assert 1<=args.workers<=16;main(args.workers)
