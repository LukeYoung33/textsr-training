"""HX52 native A VOSR2u V3 with historical0921/paired comparison."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE=DATA/'HX_testset'
OUT=DATA/'hx52_a_v3_0921_comparison_20261026'
MODELS=DATA/'v3_val583_abc_camera280_20261026/camera/models'
HISTORY=Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/v3_native_four_hx_20261015')

def worker(rank):
    """@luna-comment Input: disjoint rank0..3 HX52. Output: same-resolution A predictions and source receipts.
    Key Steps: frozen strict V3 load, native full-frame256/224 restore, byte-hash provenance.
    Key Params: original JPEG pixels, no tonal processing/OCR/warp. State/Dependencies: CUDA and own output files.
    """
    import cv2,torch,yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_vosr_six_models import restore_block
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    cv2.setNumThreads(1);torch.set_num_threads(2)
    binding=json.loads((MODELS.parent/'MODEL_BINDINGS.json').read_text())['a_v3']
    config=MODELS/'a_v3.yml';checkpoint=MODELS/'a_v3.pth'
    assert sha(config)==binding['config_sha256'] and sha(checkpoint)==binding['checkpoint_sha256']=='ad6362d77c9895546f9c3430239cec0dc2a322d58eeed5057e4e07e0e513cc41'
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    files=sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'));assert len(files)==52
    for index,path in enumerate(files):
        if index%4!=rank:continue
        image=cv2.imread(str(path));assert image is not None
        result,details=restore_block(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),model,'cuda')
        assert result.shape==image.shape
        identity=f'{index+1:03d}';target=OUT/'final'/(identity+'.png');image_write(target,cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
        save_json(OUT/'receipts'/(identity+'.json'),dict(id=identity,source=str(path),source_sha256=sha(path),source_hw=list(image.shape[:2]),binding=binding,restoration=details,final=str(target),final_sha256=sha(target),adapter_sha256=sha(__file__)))

def execute(rank):
    """@luna-comment Input: rank/GPU0..3. Output: child exit record. Key Steps: isolated log/environment.
    Key Params: one shard/GPU; four max. State/Dependencies: owned subprocesses only, old results immutable.
    """
    log=OUT/'logs'/f'{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        process=subprocess.run([sys.executable,'-u',__file__,'worker',str(rank)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    return dict(rank=rank,exit_code=process.returncode)

def run():
    """@luna-comment Input: authorized52-photo scope. Output: checked final completion after four shards.
    Key Steps: exclusive output lock and zero-exit join. Key Params: same completed A checkpoint.
    State/Dependencies: Linux flock/CUDA/NAS, isolated outputs only.
    """
    import fcntl
    from tools.evaluate_offline_camera_matrix import save_json
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=4) as pool:queue=list(pool.map(execute,range(4)))
        save_json(OUT/'QUEUE.json',queue);assert all(r['exit_code']==0 for r in queue),queue
        save_json(OUT/'COMPLETE.json',dict(passed=True,images=52))

def deliver():
    """@luna-comment Input: complete A52 and clean historical0921/paired summaries. Output: checked local viewer.
    Key Steps: source/trace/native/checkpoint correspondence, hash old outputs and new copies, four-panel render.
    Key Params: historical strategies explicit; no objective score/quality ranking or old recomputation.
    State/Dependencies: Pillow/NAS/existing renderer; isolated E results, old/source images read-only.
    """
    from PIL import Image
    from tools.build_vosr2u_local_three_viewer import digest
    from tools.build_camera_2x2_viewer import render_html
    remote=Path(OUT.as_posix().replace('/AI_DNS/','Y:/',1));history=Path(HISTORY.as_posix().replace('/AI_DNS/','Y:/',1))
    assert json.loads((remote/'COMPLETE.json').read_text())['passed']
    queue=json.loads((remote/'QUEUE.json').read_text());assert len(queue)==4 and all(r['exit_code']==0 for r in queue)
    local=ROOT/'evaluation/hx52_a_v3_0921_comparison_20261026';view=ROOT/'output/hx52_a_v3_0921_comparison_20261026'
    groups={};bindings={}
    for key,arm,expected in [('old0921','01_0921data_degraded','3d48a3c76c3c017f551b5a25708b5c607ed194fddbaaa118549a6b2844068a1a'),('paired_v3','02_displayfit_paired','95d95cc66910a5551fd5cd33b53526f9e1c25d37dc8ee23923c2ca6180b4e725')]:
        summary=json.loads((history/arm/'summary.json').read_text(encoding='utf-8'));assert summary['status']=='passed' and summary['images']==52 and summary['checkpoint_sha256']==expected
        groups[key]={Path(r['source']).name:r for r in summary['results']};assert len(groups[key])==52
        bindings[key]=dict(checkpoint=summary['checkpoint'],checkpoint_sha256=summary['checkpoint_sha256'])
    receipts=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((remote/'receipts').glob('*.json'))]
    assert len(receipts)==len(list((remote/'final').glob('*.png')))==52
    rows=[];records=[]
    for receipt in receipts:
        assert receipt['binding']['checkpoint_sha256']=='ad6362d77c9895546f9c3430239cec0dc2a322d58eeed5057e4e07e0e513cc41'
        source=Path(receipt['source'].replace('/AI_DNS/','Y:/',1));assert digest(source)==receipt['source_sha256']
        with Image.open(source) as image:w,h=image.size
        assert receipt['source_hw']==[h,w] and receipt['restoration']['canvas_hw']==[256,256] and not receipt['restoration']['ocr_cropping']
        item=dict(id=receipt['id'],name=source.name,width=w,height=h,lq=source.resolve().as_uri(),images={});audit=dict(receipt=receipt,historical={});traces=[]
        for key,group in groups.items():
            old=group[source.name];assert old['source_sha256']==receipt['source_sha256'] and old['outside_unchanged'] and old['canvas_hw']==[96,480]
            path=Path(old['output'].replace('/AI_DNS/','Y:/',1));assert digest(path)==old['output_sha256']
            with Image.open(path) as image:assert image.size==(w,h)
            item['images'][key]=path.resolve().as_uri();audit['historical'][key]=old;traces.append(old['trace_sha256'])
        assert len(set(traces))==1
        path=Path(receipt['final'].replace('/AI_DNS/','Y:/',1));assert digest(path)==receipt['final_sha256']
        with Image.open(path) as image:assert image.size==(w,h)
        target=local/'final'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        assert digest(target)==receipt['final_sha256']
        item['images']['a_v3']=target.resolve().as_uri();rows.append(item);records.append(audit)
    assert {r['name'] for r in rows}==set(groups['old0921'])==set(groups['paired_v3'])
    models=[dict(key='lq',label='LQ · HX_testset原图',note='52张原始照片，不是NV12的61张'),dict(key='old0921',label='V3 · 0921数据 · 合成退化',note='250K历史结果；原生96×480，固定OCR区域，16px贴回融合'),dict(key='paired_v3',label='V3 · 原配对GT · 文本行',note='02_displayfit_paired 250K；原生96×480，固定OCR区域，16px融合'),dict(key='a_v3',label='A组 V3 · VOSR2u · 完整损失',note='250K最终权重；256×256/步长224整幅原图推理，无OCR区域限制')]
    manifest=dict(title='HX_testset · 0921 / 配对V3 / A组VOSR2u',subtitle='52张 · 四窗同步缩放与切图',notes='原始HX_testset 52张照片，与NV12 61张目录分开。\nA组V3：VOSR2u完整损失250K，整幅原图256/224切块；不做文档warp或亮度预处理/后处理。\n0921与原配对GT V3：复用已核对源图、权重与结果哈希的历史96×480 OCR区域推理。两组使用同一固定检测记录。\n训练数据和推理覆盖方式存在差异，本页不是单因素消融实验。\nA组结果已下载本地；LQ和两组历史结果引用Y盘，需保持连接。',defaults='lq,old0921,paired_v3,a_v3',models=models,rows=rows)
    view.mkdir(parents=True,exist_ok=True);(view/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(view/'index.html').write_text(render_html(manifest),encoding='utf-8')
    (local/'DELIVERY.json').write_text(json.dumps(dict(passed=True,images=52,bindings=bindings,records=records),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,images=52,viewer=str(view/'index.html'))))

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]))
    elif sys.argv[1]=='deliver':deliver()
    else:run()
