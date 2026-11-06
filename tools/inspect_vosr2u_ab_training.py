"""Input-selected, native A/B training examples of both text polarities."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'training_vosr2u_v5_ab_polarities_20261023'
MODELS = DATA/'camera_vosr2u_v5_ab_20261023'
SOURCE = DATA/'vosr_2u_block256_v1_20261023/source_view/manifests/train.jsonl'


def select():
    """@luna-comment
    Input: immutable training manifest and LQ pixels. Output: frozen eight-parent selection/contact sheet.
    Key Steps: sample deterministic candidates, classify input luminance, require contrast and distinct identities.
    Key Params: one dark and one bright per family; dark median<100, neutral bright>175, contrast>60.
    State/Dependencies: OpenCV/NumPy; model results never participate in selection; isolated NAS output.
    """
    import cv2
    import numpy as np
    from tools.evaluate_offline_camera_matrix import sha, save_json
    rows = [json.loads(x) for x in SOURCE.read_text().splitlines()]
    chosen, identities = {}, set()
    families = sorted({r['dataset'] for r in rows})
    candidates = []
    for family in families:
        group = [r for r in rows if r['dataset']==family]
        candidates.extend(group[::17]+[r for i,r in enumerate(group) if i%17])
    for row in candidates:
        family = row['dataset']
        if all((family,p) in chosen for p in ('dark','bright')):
            continue
        if row['source_identity'] in identities:
            continue
        image = cv2.imread(row['files']['lq'])
        assert image is not None
        small = cv2.resize(image,(320,320))
        gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        low,med,high = np.percentile(gray,[5,50,95]).tolist()
        chroma = np.median(small.max(axis=2).astype(float)-small.min(axis=2))
        polarity = 'dark' if med<100 else ('bright' if med>175 and chroma<35 and low<130 else None)
        if polarity is None or high-low<60 or (family,polarity) in chosen:
            continue
        assert sha(row['files']['lq'])==row['output_sha256']['lq']
        entry = dict(row=row,polarity=polarity,input_percentiles=[low,med,high])
        chosen[family,polarity]=entry
        identities.add(row['source_identity'])
        print(f'selected {family} {polarity} {row["sample_id"]}',flush=True)
    assert len(chosen)==2*len(families),list(chosen)
    selection = [chosen[key] for key in sorted(chosen)]
    sheet = np.full((len(selection)//2*260,720,3),245,np.uint8)
    for index,entry in enumerate(selection):
        entry['id']=f'{index+1:03d}'
        image=cv2.imread(entry['row']['files']['lq'])
        h,w=image.shape[:2]
        thumb=cv2.resize(image,(round(w*min(350/w,225/h)),round(h*min(350/w,225/h))))
        y,x=index//2*260,index%2*360
        sheet[y:y+thumb.shape[0],x:x+thumb.shape[1]]=thumb
        cv2.putText(sheet,entry['id']+' '+entry['row']['dataset']+' '+entry['polarity'],(x+5,y+248),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    OUT.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(OUT/'selection.jpg'),sheet)
    save_json(OUT/'selection.json',dict(source_manifest_sha256=sha(SOURCE),selection_basis='LQ only, before inference',samples=selection))


def worker(arm):
    """@luna-comment
    Input: frozen LQ-selected parents and exact final A/B snapshots. Output: native full-image PNGs and receipts.
    Key Steps: verify source/model hashes, strict load, reuse RGB256/224 block restoration, preserve native size.
    Key Params: no preprocessing, OCR crop, warp or tonal postprocessing; GT never enters model.
    State/Dependencies: existing CUDA block helper and immutable snapshots; writes only isolated arm files.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_vosr_six_models import restore_block
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    checkpoint,config=MODELS/'models'/(arm+'.pth'),MODELS/'models'/(arm+'.yml')
    binding=json.loads((MODELS/'MODEL_BINDINGS.json').read_text())[arm]
    assert sha(checkpoint)==binding['checkpoint_sha256'] and sha(config)==binding['config_sha256']
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    selection=json.loads((OUT/'selection.json').read_text())
    for entry in selection['samples']:
        row=entry['row']
        assert {k:sha(v) for k,v in row['files'].items()}==row['output_sha256']
        image=cv2.imread(row['files']['lq'])
        target=cv2.imread(row['files']['gt'])
        assert image is not None and target is not None and image.shape==target.shape
        output,details=restore_block(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),model,'cuda')
        output=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
        assert output.shape==image.shape and output.dtype==np.uint8
        path=OUT/arm/(entry['id']+'.png')
        image_write(path,output)
        save_json(OUT/arm/(entry['id']+'.json'),dict(id=entry['id'],sample_id=row['sample_id'],source_hashes=row['output_sha256'],binding=binding,final_sha256=sha(path),shape=list(output.shape),restoration=details,selection_sha256=sha(OUT/'selection.json'),tool_sha256=sha(__file__)))
        print(arm,entry['id'],row['sample_id'],flush=True)
    assert sha(checkpoint)==binding['checkpoint_sha256']


def publish():
    """@luna-comment
    Input: frozen selection and completed arm receipts. Output: checked LQ/GT/A/B file-open viewer.
    Key Steps: verify every asset hash/dimension and shared source binding; reuse synchronized native viewer.
    Key Params: training examples only, new VOSR2u GT, no metric report or generalization claim.
    State/Dependencies: read-only sources/models and generic renderer; isolated HTML/manifest/audit.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import file_url,render_html
    from tools.evaluate_offline_camera_matrix import sha,save_json
    selection=json.loads((OUT/'selection.json').read_text())
    definitions=[('lq','LQ · 训练输入'),('gt','GT · 最新 VOSR2u 训练目标'),('a_v5','A组 V5 · 原始权重起训 · 250K'),('b_v5','B组 V5 · 上一版分块权重起训 · 250K')]
    manifest=dict(title='训练集 · 暗底 / 亮底 · V5 A/B',subtitle='LQ / GT / A组 / B组 · 同步缩放与切图',notes='各来源暗底、亮底各选1张；按输入选图，模型结果不参与筛选。\n原尺寸整图256切块、步长224、块间融合；无预处理、无文字行裁剪、无亮度后处理。\nGT为最新VOSR2u训练目标（已按训练规则匹配LQ尺寸）。这些是训练样本，不能代表验证集泛化表现。',defaults='lq,gt,a_v5,b_v5',models=[dict(key=k,label=v,note='已对齐训练整图 · 原始尺寸') for k,v in definitions],rows=[])
    for entry in selection['samples']:
        row=entry['row']
        paths=dict(lq=Path(row['files']['lq']),gt=Path(row['files']['gt']),a_v5=OUT/'A'/(entry['id']+'.png'),b_v5=OUT/'B'/(entry['id']+'.png'))
        with Image.open(paths['lq']) as image:width,height=image.size
        for key,path in paths.items():
            with Image.open(path) as image:assert image.size==(width,height)
            if key in ('lq','gt'):assert sha(path)==row['output_sha256'][key]
            else:
                receipt=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==receipt['final_sha256'] and receipt['source_hashes']==row['output_sha256']
        label='暗底亮字' if entry['polarity']=='dark' else '亮底暗字'
        manifest['rows'].append(dict(id=entry['id'],name=label+' · '+row['dataset']+' · '+row['sample_id'],width=width,height=height,lq=file_url(paths['lq']),images={k:file_url(p) for k,p in paths.items() if k!='lq'}))
    save_json(OUT/'manifest.json',manifest)
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    save_json(OUT/'AUDIT.json',dict(passed=True,samples=len(manifest['rows']),finals=2*len(manifest['rows']),selection_sha256=sha(OUT/'selection.json'),metrics='not_requested',training_examples=True))


if __name__=='__main__':
    if sys.argv[1]=='select':select()
    elif sys.argv[1]=='worker':worker(sys.argv[2])
    elif sys.argv[1]=='publish':publish()
    else:raise ValueError(sys.argv[1])
