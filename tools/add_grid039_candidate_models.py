"""Add old/new final500K model outputs to existing candidate GT review."""
from pathlib import Path
import argparse
import json
import sys
import shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'grid039_target_candidates_20261104'
LABELS = {'teacher_l1':'V3M · VOSR GT · L1-only · 500K','original_dilate2_l1':'V3M · 原始GT＋2px · L1-only · 500K'}


def file_url(path):
    """@luna-comment Input:ownedNASimagepath. Output:equivalentmappedYfileURL. Key Steps:strictNASprefix/URLencoding without importing historical runtime. Key Params:nootherroots. State/Dependencies:urllib only."""
    from urllib.parse import quote
    value = str(path)
    assert value.startswith('/AI_DNS/'),value
    return 'file:///Y:/'+quote(value[len('/AI_DNS/'):],safe='/')


def image_path(url):
    """@luna-comment Input:mappedYfileURL. Output:originalNAS path. Key Steps:strictprefix andpercentdecode. Key Params:noarbitraryrootfallback. State/Dependencies:urllib only."""
    from urllib.parse import unquote
    prefix = 'file:///Y:/'
    assert url.startswith(prefix),url
    return DATA.parent.parent.parent/unquote(url[len(prefix):])


def infer():
    """@luna-comment Input:existing19parent145patchinventories and2final500Kmodels. Output:618newrawPNGs andhashboundinventory. Key Steps:strictload,acceptedfullfusion,directsinglepatch,sameXYcrops,pixelidentitygate. Key Params:nopost,nomodelupdates;existingGTnotrechecked. State/Dependencies:oneownedCUDAprocess andnewoutputpaths."""
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.restore_crop8_block256 import restore_block
    from tools.diagnose_grid039 import decode
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    parents = json.loads((OUT/'PARENTS.json').read_text())['rows']
    patches = json.loads((OUT/'PATCHES.json').read_text())['rows']
    bindings = dict(teacher_l1=json.loads((DATA/'latest_six_500k_mojito280_20261103/MODEL_BINDINGS.json').read_text())['v3m_l1'],
                    original_dilate2_l1=json.loads((DATA/'v3m_original_dilate2_l1_mojito280_20261104/MODEL_BINDINGS.json').read_text())['v3m_dilate2_l1'])
    outputs = dict(parents={},patches={})
    receipts = []
    for arm,binding in bindings.items():
        assert digest(binding['checkpoint'])==binding['checkpoint_sha256'] and digest(binding['config'])==binding['config_sha256']
        model = build_network(yaml.safe_load(Path(binding['config']).read_text())['network_g']).cuda().eval()
        model.load_state_dict(torch.load(binding['checkpoint'],map_location='cpu')['params'],strict=True)
        for number,parent in enumerate(parents,1):
            source = image_path(parent['lq'])
            rgb = cv2.cvtColor(cv2.imread(str(source)),cv2.COLOR_BGR2RGB)
            assert rgb.shape[:2]==(parent['height'],parent['width'])
            result,details = restore_block(rgb,model,'cuda')
            directory = OUT/'model_results'/arm/'parents'
            directory.mkdir(parents=True,exist_ok=True)
            target = directory/(parent['id']+'.png')
            assert cv2.imwrite(str(target),cv2.cvtColor(result,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
            outputs['parents'].setdefault(parent['id'],{})[arm] = file_url(target)
            receipts.append(dict(parent=parent['id'],arm=arm,kind='full',output=str(target),sha256=digest(target),input_sha256=digest(source),restoration=details))
            children = [r for r in patches if r['parent']==parent['id']]
            for start in range(0,len(children),8):
                batch = children[start:start+8]
                inputs = []
                for row in batch:
                    value = cv2.cvtColor(cv2.imread(str(image_path(row['lq']))),cv2.COLOR_BGR2RGB)
                    x,y = row['metrics']['xy']
                    assert np.array_equal(value,rgb[y:y+256,x:x+256]),row['id']
                    inputs.append(value)
                singles = decode(model,inputs)
                for row,single in zip(batch,singles):
                    x,y = row['metrics']['xy']
                    single = np.clip(np.rint(single),0,255).astype(np.uint8)
                    fused = result[y:y+256,x:x+256]
                    for suffix,value in [('single',single),('fullcrop',fused)]:
                        assert value.shape==(256,256,3)
                        folder = OUT/'model_results'/arm/suffix
                        folder.mkdir(parents=True,exist_ok=True)
                        target = folder/(row['id']+'.png')
                        assert cv2.imwrite(str(target),cv2.cvtColor(value,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
                        outputs['patches'].setdefault(row['id'],{})[arm+'_'+suffix] = file_url(target)
                        receipts.append(dict(patch=row['id'],parent=parent['id'],arm=arm,kind=suffix,output=str(target),sha256=digest(target)))
            save(OUT/'MODEL_PROGRESS.json',dict(arm=arm,parents_done=number,parents_per_arm=19,new_outputs=len(receipts),total_new_outputs=618))
            print('MODEL',arm,'PARENT',number,'OUTPUTS',len(receipts),flush=True)
        assert digest(binding['checkpoint'])==binding['checkpoint_sha256']
        del model
    assert len(receipts)==618 and len(outputs['patches'])==145 and len(outputs['parents'])==19
    save(OUT/'MODEL_BINDINGS.json',bindings)
    save(OUT/'MODEL_OUTPUTS.json',outputs)
    save(OUT/'MODEL_AUDIT.json',dict(passed=True,parents=19,patches=145,new_outputs=618,existing_gt_rechecked=False,records=receipts))
    save(OUT/'MODEL_COMPLETE.json',dict(passed=True,new_outputs=618))


def publish():
    """@luna-comment Input:new618modeloutputaudit/existinglocalGTviewer. Output:patch12option/parent6optionpages withnewmodelsvisible. Key Steps:appendnewpaths withoutoldimagechecks,renderacceptedsharedview,updateoverviewlinks. Key Params:singleversusfusedcropclearlylabeled,nopost. State/Dependencies:localmetadata andnewNASmodelimages."""
    from tools.build_camera_2x2_viewer import render_html
    source = Path('Y:/guang.yang/TextSR/data/grid039_target_candidates_20261104')
    destination = ROOT/'evaluation/grid039_target_candidates_20261104'
    assert json.loads((source/'MODEL_COMPLETE.json').read_text())['passed']
    outputs = json.loads((source/'MODEL_OUTPUTS.json').read_text())
    patches = json.loads((destination/'patch_manifest.json').read_text(encoding='utf-8'))
    parents = json.loads((destination/'parent_manifest.json').read_text(encoding='utf-8'))
    patch_keys = {arm+'_'+suffix for arm in LABELS for suffix in ['single','fullcrop']}
    patches['models'] = [m for m in patches['models'] if m['key'] not in patch_keys]
    for suffix,description in [('single','单块256直接输出 · 无融合'),('fullcrop','整图融合后同位置裁块')]:
        for arm,label in LABELS.items():
            patches['models'].append(dict(key=arm+'_'+suffix,label=label+' | '+description,note='模型原始输出，未加后处理；单块含边界响应，整图crop8/smooth16融合'))
    for row in patches['rows']:
        row['images'].update(outputs['patches'][row['id']])
    patches['defaults'] = 'lq,vosr_aligned,dilate2_aligned,background_mask,teacher_l1_single,original_dilate2_l1_single,teacher_l1_fullcrop,original_dilate2_l1_fullcrop'
    patches['subtitle'] = '145个训练patch · GT与两组模型 · 12个选项'
    note = '\n新增两组最终500K模型原始输出：VOSR GT L1-only、原始GT＋2px L1-only。单块结果直接处理256输入，不裁边／融合；整图裁块来自256/224/crop8/smooth16的整图推理，同XY提取。均未加后处理。这些是实际训练样本，用于诊断，不代表独立泛化验证。'
    if note not in patches['notes']:
        patches['notes'] += note
    page = render_html(patches).replace('for(let i=0;i<4;i++)','for(let i=0;i<8;i++)').replace(
         'grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    (destination/'patches.html').write_text(page,encoding='utf-8')
    parents['models'] = [m for m in parents['models'] if m['key'] not in LABELS]
    for arm,label in LABELS.items():
        parents['models'].append(dict(key=arm,label=label+' | 整图原始输出',note='256/224/batch8/crop8/smooth16；未加后处理'))
    for row in parents['rows']:
        row['images'].update(outputs['parents'][row['id']])
    parents['defaults'] = 'lq,vosr_raw,original_raw,dilate2_raw,teacher_l1,original_dilate2_l1'
    parents['subtitle'] = '19张共同训练图 · 四版数据＋两组模型'
    if note not in parents['notes']:
        parents['notes'] += note
    page = render_html(parents).replace('for(let i=0;i<4;i++)','for(let i=0;i<6;i++)').replace(
         'grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    (destination/'parents.html').write_text(page,encoding='utf-8')
    save(destination/'patch_manifest.json',patches)
    save(destination/'parent_manifest.json',parents)
    text = (destination/'index.html').read_text(encoding='utf-8')
    text = text.replace('整图四版本','整图四版数据＋两组模型').replace('八版本对比','GT与模型对比（12个选项）')
    notice = '<p><b>已加入两组最终500K模型输出。</b>Patch页同时展示单块输出和整图融合后裁块；整图页同时展示四版数据与两组模型。</p>'
    if notice not in text:
        text = text.replace('<div class="grid">',notice+'<div class="grid">',1)
    (destination/'index.html').write_text(text,encoding='utf-8')
    for filename in ['MODEL_BINDINGS.json','MODEL_AUDIT.json','MODEL_COMPLETE.json']:
        shutil.copy2(source/filename,destination/filename)
    print(json.dumps(dict(passed=True,parents=19,patches=145,parent_options=6,patch_options=12,new_outputs=618)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['infer','publish'])
    args = parser.parse_args()
    infer() if args.mode=='infer' else publish()
