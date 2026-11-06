"""Create a self-contained fixed32 contrast comparison and verified ZIP."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mojito_s22_thinning import digest,save
from tools.native_y_curve_four_images import process_pair,CurveTrial

DATA=Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE=DATA/'mojito32_s22_thin_blend_20261104'
OUT=DATA/'mojito32_s22_thin_blend_portable_20261105'
PACKAGE=OUT/'Mojito32_Compare'


def process(job):
    """@luna-comment Input:row/key/source receipt. Output:raw andcurrentpostPNG/hash metadata. Key Steps:sourcehash,copyraw,currentcurve zero-trial,roundtripcheck. Key Params:.75/2/24,noUSM. State/Dependencies:ownedpackage;immutableinputs."""
    cv2.setNumThreads(1)
    row,key,entry=job
    source=SOURCE/entry['path']
    assert digest(source)==entry['sha256']
    image=cv2.imread(str(source))
    assert image.shape==(row['height'],row['width'],3)
    raw=PACKAGE/'images/raw'/key/(row['id']+'.png')
    post=PACKAGE/'images/post'/key/(row['id']+'.png')
    raw.parent.mkdir(parents=True,exist_ok=True);post.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,raw)
    value,zero,stats=process_pair(image,CurveTrial(16.,48.,0.,0.))
    assert np.array_equal(value,zero)
    assert np.abs(value.astype(np.int16)-image.astype(np.int16)).max()<=24
    assert cv2.imwrite(str(post),value,[cv2.IMWRITE_PNG_COMPRESSION,3])
    assert np.array_equal(cv2.imread(str(post)),value)
    return dict(id=row['id'],key=key,width=row['width'],height=row['height'],raw=str(raw.relative_to(PACKAGE)),
                post=str(post.relative_to(PACKAGE)),raw_sha256=digest(raw),post_sha256=digest(post),
                changed_pixels=stats['changed_pixels_current_from_source'],clipped_channels=stats['rgb_clipped_channels_current'])


def publish(rows,records):
    """@luna-comment Input:32rows/320rawpostrecords,localtemplate. Output:portableHTML andsanitizedmanifest. Key Steps:relativeURLdata,20options,rawpostbuttons,README. Key Params:2x5panels,defaultpost. State/Dependencies:allassets beneathPACKAGE,no network/runtime."""
    mapping={(r['id'],r['key']):r for r in records}
    title='Mojito32 · 中间形态 · 对比度增强'
    subtitle='上排：−1px GT；下排：−2px GT<br>每排：输入 → 25% → 50% → 75% → GT'
    notes='先做RGB插值，再统一进行局部对比度增强（增益0.75、死区2、软限幅24；无USM）。<br>增强后的比例标签指增强前的GT占比，增强后不再严格线性。<br>使用上方按钮切换整组处理前/增强后；各窗口也可单独选择。<br>缩窄量为笔画总宽度名义值。全图原生尺寸；不代表真实相机PSF。<br>此包完全离线，解压后打开index.html；保留images文件夹相对位置。'
    models=[]
    labels={'v3_input':'V3输入 · GT 0%','original_gt':'未缩窄 S2.2 GT'}
    for arm,width in [('minus1',1),('minus2',2)]:
        labels[arm]=f'−{width}px GT · GT 100%'
        for a in (25,50,75): labels[arm+f'_a{a}']=f'−{width}px路线 · GT {a}%'
    for key,label in labels.items():
        for post in (False,True):
            k=('lq' if key=='v3_input' else key) if not post else key+'_post'
            models.append(dict(key=k,label=label+('｜增强后' if post else '｜处理前'),note='固定局部对比度0.75/2/24' if post else '原始融合/端点，无追加后处理'))
    defaults=[]
    for arm in ('minus1','minus2'):
        defaults += ['v3_input_post',arm+'_a25_post',arm+'_a50_post',arm+'_a75_post',arm+'_post']
    display=[]
    for row in rows:
        images={}
        for key in labels:
            record=mapping[row['id'],key]
            if key!='v3_input':images[key]=record['raw']
            images[key+'_post']=record['post']
        display.append(dict(id=row['id'],name=row['name'],width=row['width'],height=row['height'],lq=mapping[row['id'],'v3_input']['raw'],images=images))
    data=dict(title=title,models=models,defaults=','.join(defaults),rows=display)
    text=(OUT/'template.html').read_text(encoding='utf-8')
    text=re.sub(r'const DATA=.*?;</script>',lambda _:'const DATA='+json.dumps(data,ensure_ascii=False)+';</script>',text,count=1,flags=re.S)
    text=re.sub(r'<title>.*?</title>','<title>'+title+'</title>',text,count=1)
    text=re.sub(r'<aside><h1>.*?</small>','<aside><h1>'+title+'</h1><small>'+subtitle+'</small>',text,count=1,flags=re.S)
    text=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+notes+'</div>',text,count=1,flags=re.S)
    text=text.replace('<div class="toolbar">','<div class="toolbar"><button id="showRaw">整组：处理前</button><button id="showPost">整组：增强后</button>',1)
    text=text.replace('href="RESULTS.json"','href="MANIFEST.json"')
    text+='''<script>
// @luna-comment Input:raw/post mode. Output:allpanels andSpace reference switched. Key Steps:map modelkeys,loadpanels/updatehash. Key Params:tenexisting slots. State/Dependencies:inlineviewer only.
function setContrastMode(post){for(const p of panels){let k=p.select.value;if(k==='lq')k='v3_input';k=k.replace(/_post$/,'');p.select.value=post?k+'_post':(k==='v3_input'?'lq':k);loadPanel(p);}$('flashTarget').value=post?'v3_input_post':'lq';loadComparison();updateHash();}
$('showRaw').onclick=()=>setContrastMode(false);$('showPost').onclick=()=>setContrastMode(true);
</script>'''
    assert 'file:///' not in text and '/AI_DNS/' not in text and 'Y:/' not in text
    (PACKAGE/'index.html').write_text(text,encoding='utf-8')
    save(PACKAGE/'MANIFEST.json',dict(samples=32,files=640,post='current local Y contrast gain0.75 deadband2 softcap24; after blending; PNG RGB approximation',records=records))
    (PACKAGE/'README.txt').write_text('Mojito32 portable comparison\n\nExtract the entire ZIP, then double-click index.html in a modern browser.\nKeep the images folder beside index.html. No server, Python, NAS or internet is needed.\nTop row: nominal stroke-width -1px target. Bottom row: -2px target.\nColumns: V3 input, GT25%, GT50%, GT75%, GT100%.\nUse the two leading toolbar buttons to switch raw/current contrast for all panels.\nWheel/pan/ROI are synchronized. Space temporarily shows the selected reference.\nContrast runs AFTER RGB interpolation, so postprocessed fractions are labels for their input blends, not exact postprocessed linear fractions.\nAll source images are native-resolution lossless PNG. This is an experimental target ladder, not a physical camera degradation model.\n',encoding='utf-8')


def main():
    """@luna-comment Input:verified320images andviewer template. Output:640image portableZIP,CRC/hash audit. Key Steps:8CPUpost,relativecontainment/SHAverify,ZIPstore/testzip. Key Params:ownedOUT,no sourcewrites. State/Dependencies:NAS space,Pythonzip64."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);PACKAGE.mkdir(exist_ok=True)
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=json.loads((SOURCE/'RESULTS.json').read_text());rows=result['rows'];assert result['passed'] and len(rows)==32
    records=[];start=time.time()
    with ProcessPoolExecutor(8) as pool:
        jobs=[pool.submit(process,(r,k,v)) for r in rows for k,v in r['outputs'].items()]
        for future in as_completed(jobs):
            records.append(future.result());save(OUT/'STATUS.json',dict(stage='contrast',completed=len(records),total=320,seconds=time.time()-start))
    records.sort(key=lambda r:(r['id'],r['key']))
    publish(rows,records)
    for r in records:
        for mode in ('raw','post'):
            p=(PACKAGE/r[mode]).resolve();assert PACKAGE.resolve() in p.parents
            assert digest(p)==r[mode+'_sha256']
    archive=OUT/'Mojito32_Compare.zip'
    save(OUT/'STATUS.json',dict(stage='packaging',completed=320,total=320,seconds=time.time()-start))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(PACKAGE.rglob('*')):
            if path.is_file():z.write(path,str(Path(PACKAGE.name)/path.relative_to(PACKAGE)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.endswith('.png')])==640
    save(OUT/'COMPLETE.json',dict(passed=True,samples=32,raw=320,post=320,files=640,archive_bytes=archive.stat().st_size,
         archive_sha256=digest(archive),core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'),seconds=time.time()-start))
    print((OUT/'COMPLETE.json').read_text(),flush=True)


if __name__=='__main__':main()
