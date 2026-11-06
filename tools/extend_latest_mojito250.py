"""Append a verified historical V3M250K raw/post reference to the current gallery."""
from pathlib import Path
import sys,json,time,os,argparse,fcntl
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');CURRENT=DATA/'latest_six_500k_mojito280_20261103'
CHOICES={'nocp':('s22_guard_contrast_four_mojito280_20261030','nocp_nosdf','V3M 历史250K · guard＋对比度对齐＋noCP'),'original':('v3pre_three_mojito280_20261028','full','V3M 历史250K · VOSR2u＋V3预处理 Full')}


def verify(job):
    """@luna-comment Input:current/historical same-ID rows and frozen binding. Output:verified raw receipt/path metadata. Key Steps:input identity,checkpoint binding,raw hash andnative geometry. Key Params:exact250K,crop8/smooth16. State/Dependencies:read-only historical archive."""
    from PIL import Image
    entry,old,binding,archive,key=job
    assert entry['id']==old['id'] and all(entry[k]==old[k] for k in ['input_sha256','source_sha256','stored_shape','orientation','width','height'])
    path=Path(archive)/'results'/key/'final'/(entry['id']+'.png');receipt=Path(archive)/'results'/key/'receipts'/(entry['id']+'.json');r=json.loads(receipt.read_text())
    assert r['binding']==binding and r['input_sha256']==entry['input_sha256'] and digest(path)==r['final_sha256']
    d=r['restoration'];assert d['stitch_policy']=='crop8_smooth16' and d['crop_pixels']==8 and d['blend_pixels']==16
    assert d['canvas_hw']==[256,256] and d['stride_hw']==[224,224] and not d['ocr_cropping']
    with Image.open(path) as im:assert im.size==(entry['width'],entry['height'])
    return dict(id=entry['id'],raw=str(path),raw_sha256=r['final_sha256'],receipt_sha256=digest(receipt))


def materialize(job):
    """@luna-comment Input:owned reference root/arm/sample. Output:accepted frozen current-post result. Key Steps:process-local rebind then delegate same post implementation. Key Params:.75/2/24. State/Dependencies:CPU worker,immutable raw symlink."""
    folder,key,entry=job
    import tools.evaluate_latest_six_mojito280 as core
    core.OUT=Path(folder);core.post((key,entry))


def main(choice):
    """@luna-comment Input:resolved historical recipe. Output:280 audited post references andtwo newviewer options. Key Steps:lock,verify originals,parallelpost,append manifest,rerender local later. Key Params:do not replace six defaultpanels. State/Dependencies:independent reference audit andowned metadata."""
    from tools.build_camera_2x2_viewer import file_url
    archive,key,label=CHOICES[choice];archive=DATA/archive;arm='previous250_'+choice;folder=CURRENT/'references'/arm;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((CURRENT/'COMPLETE.json').read_text())['passed'] and json.loads((archive/'AUDIT.json').read_text())['passed']
    binding=json.loads((archive/'MODEL_BINDINGS.json').read_text())[key]
    assert binding.get('iteration',binding.get('pretrain_iter'))==250000
    for name in ['config','checkpoint']:assert digest(binding[name])==binding[name+'_sha256']
    assert binding['stitcher_sha256']==digest(ROOT/'tools/restore_crop8_block256.py')
    save(folder/'BINDING.json',dict(choice=choice,label=label,binding=binding,source_audit_sha256=digest(archive/'AUDIT.json')))
    samples=json.loads((CURRENT/'selection.json').read_text())['samples'];olds={e['id']:e for e in json.loads((archive/'selection.json').read_text())['samples']};assert len(samples)==len(olds)==280
    (folder/'results').mkdir(exist_ok=True);link=folder/'results'/arm
    if not link.exists():link.symlink_to(archive/'results'/key,target_is_directory=True)
    assert link.resolve()==(archive/'results'/key).resolve();records=[];started=time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n,row in enumerate(pool.map(verify,[(e,olds[e['id']],binding,str(archive),key) for e in samples]),1):
            records.append(row)
            if n%20==0:save(folder/'STATUS.json',dict(phase='verify_raw',completed=n,total=280))
    historical={r['id']:r for r in json.loads((archive/'AUDIT.json').read_text())['records']}
    for r in records:
        prior=historical[r['id']]
        expected=prior['outputs'][key] if 'outputs' in prior else prior['models'][key]
        assert r['raw_sha256']==expected.get('final_sha256',expected.get('sha256')) and r['receipt_sha256']==expected['receipt_sha256']
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n,_ in enumerate(pool.map(materialize,[(str(folder),arm,e) for e in samples]),1):
            elapsed=time.time()-started;save(folder/'STATUS.json',dict(phase='postprocessing',completed=n,total=280,elapsed_seconds=elapsed,eta_seconds=elapsed/n*(280-n)))
    manifest=json.loads((CURRENT/'manifest.json').read_text());rows={r['id']:r for r in manifest['rows']};audits=[]
    core=digest(ROOT/'tools/native_y_curve_four_images.py')
    for r in records:
        receipt=folder/'post'/arm/'receipts'/(r['id']+'.json');p=json.loads(receipt.read_text());path=folder/'post'/arm/'final'/(r['id']+'.png')
        assert p['source_sha256']==r['raw_sha256'] and p['core_sha256']==core and digest(path)==p['output_sha256']
        rows[r['id']]['images'][arm]=file_url(path);rows[r['id']]['images'][arm+'_raw']=file_url(r['raw'])
        audits.append(dict(**r,post_sha256=p['output_sha256']))
    for suffix,caption in [('', '固化后处理'),('_raw','模型原始输出')]:
        item=dict(key=arm+suffix,label=label+' | '+caption,note='历史最终250K；相同V3缓存与crop8/smooth16；'+('后处理 .75/2/24' if not suffix else '未加后处理'))
        manifest['models']=[m for m in manifest['models'] if m['key']!=item['key']]+[item]
    manifest['title']='Mojito280 · 六模型500K＋历史V3M250K';manifest['subtitle']='六组最终500K＋历史250K参照'
    note='\n历史参照：'+label+'；下拉菜单同时保留原始输出和相同固化后处理。训练数据、策略和轮数均有差异，作为历史参照。'
    if note not in manifest['notes']:manifest['notes']+=note
    for name in ['manifest.json','index.html']:
        p=CURRENT/name;backup=CURRENT/(p.stem+'.before250_'+choice+p.suffix)
        if not backup.exists():backup.write_bytes(p.read_bytes())
    save(folder/'AUDIT.json',dict(passed=True,images=280,post_outputs=280,records=audits));save(CURRENT/'manifest.json',manifest)
    save(folder/'COMPLETE.json',dict(passed=True,images=280,choice=choice));save(folder/'STATUS.json',dict(phase='complete',completed=280,total=280))
    print('complete',arm,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--choice',choices=list(CHOICES),default='nocp');args=parser.parse_args();main(args.choice)
