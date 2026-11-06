"""Build a file-open comparison viewer from immutable, source-bound camera results."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_camera_compare_missing import DATA, OLD, OUT, INV, ARMS, model_paths


def digest(path):
    """@luna-comment Input:asset path. Output:SHA256. Key Steps:stream chunks. Key Params:4MiB. State/Dependencies:read-only files."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            result.update(chunk)
    return result.hexdigest()


def file_url(path):
    """@luna-comment Input:approved remote NAS path. Output:Windows mapped-Y file URL. Key Steps:replace exact mount prefix and URL encode. Key Params:reject other roots. State/Dependencies:client Y mapping."""
    value = str(path)
    prefix = '/AI_DNS/'
    if not value.startswith(prefix):
        raise ValueError('Not a mapped NAS asset: '+value)
    return 'file:///Y:/'+quote(value[len(prefix):], safe='/')


def models():
    """@luna-comment Input:fixed strategy matrix. Output:ten model definitions plus LQ/pending GT. Key Steps:explicit labels and historical geometry caveats. Key Params:V2_2 only for pre columns. State/Dependencies:manifest metadata."""
    rows = [dict(key='lq', label='LQ · 原始图像', note='原始相机图像 · 未增强'), dict(key='gt', label='GT · VOSR（待补充）', note='暂未生成；不以其他结果代替', pending=True)]
    for family, title, variant in [('original', '原始 GT 训练', 'pre'), ('original', '原始 GT 训练', 'raw'), ('vosr', 'VOSR GT 训练', 'pre'), ('vosr', 'VOSR GT 训练', 'raw'), ('vosr', 'VOSR GT 训练', 'block')]:
        for arch in ('v3', 'v5'):
            key = f'{family}_{arch}_{variant}'
            label = {'pre':'V2_2 预处理', 'raw':'无预处理 · 文本区域', 'block':'无预处理 · 256×256 切块'}[variant]
            note = '文档检测 → warp → '+('V2_2 → 模型 → 亮度后处理 → ' if variant=='pre' else '模型 → ')+'贴回 → unwarp'
            if family == 'original' and variant == 'raw':
                note += ' · 早期 displayfit 训练；'+('整行归一到高96' if arch=='v5' else '原生96×480分块')
            else:
                note += ' · '+('整幅warp切256 / 步长224' if variant=='block' else '原生96×480分块')
            rows.append(dict(key=key, label=arch.upper()+' · '+title+' · '+label, note=note))
    rows.insert(1,dict(key='pre',label='V2_2 · 仅预处理',note='文档检测 → warp → V2_2 预处理 → unwarp贴回；不经过文字模型和亮度后处理'))
    return rows


def render_html(manifest):
    """@luna-comment Input:viewer manifest. Output:inline file-open HTML. Key Steps:escape headings/notes, embed JSON and shared geometry. Key Params:no external JS/fetch. State/Dependencies:repository template; shared by280 and HX viewers."""
    template = (ROOT/'tools/camera_2x2_viewer.html').read_text(encoding='utf-8')
    notes = manifest.get('notes','有预处理：V2_2 + 亮度后处理。\n无预处理：关闭亮度预处理和后处理。\n文档检测、warp、贴回保持一致。\n\n原始 GT 无预处理来自早期 displayfit 训练，训练批次及增强也有差异，不能视为仅预处理不同的消融实验。\nGT（VOSR）后续补充。')
    return template.replace('__TITLE__',html.escape(manifest.get('title','文字增强 · 280 张'))).replace('__SUBTITLE__',html.escape(manifest.get('subtitle','V3 / V5 · 五类训练策略\n四窗独立选模型，同步看同一位置')).replace('\n','<br>')).replace('__NOTES__',html.escape(notes).replace('\n','<br>')).replace('__MANIFEST__',json.dumps(manifest,ensure_ascii=False).replace('</','<\\/')).replace('__MATH__',(ROOT/'tools/camera_2x2_math.js').read_text(encoding='utf-8'))


def build(args):
    """@luna-comment
    Input:complete immutable results and viewer sources. Output:inline HTML/manifest/provenance audit.
    Key Steps:verify checkpoint identities, receipt/source binding, decoded dimensions and new final hashes; map URLs.
    Key Params:280 production images; development limit explicitly marked. State/Dependencies:read-only existing four-column acceptance, Pillow headers; no source mutation.
    """
    from PIL import Image
    import yaml
    six = DATA/'vosr_final_evaluation_20261022'
    reused = {'vosr_v3_raw':'04_v3_raw', 'vosr_v5_raw':'03_v5_raw', 'vosr_v3_block':'06_v3_block256_raw', 'vosr_v5_block':'05_v5_block256_raw'}
    bindings = {}
    for key in ARMS:
        config, checkpoint = model_paths(key)
        bindings[key] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), config_sha256=digest(config))
    for key, arm in reused.items():
        train = (DATA/'vosr_fullimage_block256_v1_20261021' if 'block' in key else DATA/'adaptive_vosr_preprocessed_augmented_v1_20261021')/'training_comparison_20261021'
        config = train/'configs'/(arm+'.yml')
        checkpoint = train/'experiments'/yaml.safe_load(config.read_text())['name']/'models/net_g_250000.pth'
        bindings[key] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), config_sha256=digest(config))
    rows, records = [], []
    inventory = json.loads(INV.read_text())['images']
    assert len(inventory) == 280
    for row in inventory[:args.limit or None]:
        identity = row['id']
        source = OLD/'evaluation/camera_final_20261021/source'/(identity+Path(row['name']).suffix.lower())
        assert digest(source) == row['sha256']
        # Decode orientation once: EXIF-oriented camera dimensions differ from JPEG headers for045.
        import cv2
        decoded = cv2.imread(str(source))
        height, width = decoded.shape[:2]
        output = dict(id=identity, name=row['name'], width=width, height=height, lq=file_url(source), images={})
        for key in bindings:
            base = OUT if key in ARMS else six
            receipt_file = OUT/'results'/key/'receipts'/(identity+'.json') if key in ARMS else six/'camera'/reused[key]/'receipts'/(identity+'.json')
            receipt = json.loads(receipt_file.read_text())
            assert receipt['source_sha256'] == row['sha256'] and receipt['outside_owned_unchanged']
            assert all(receipt['binding'][k] == v for k,v in bindings[key].items())
            if key.endswith('_pre'):
                assert receipt['binding']['preprocessing'] == 'V2_2'
            final = base/receipt['final']
            with Image.open(final) as image:
                assert image.size == (width, height)
            # Existing SIX finals have a completed independent acceptance. New finals are fully rehashed here.
            if key in ARMS:
                assert digest(final) == receipt['final_sha256']
            output['images'][key] = file_url(final)
            records.append(dict(id=identity, model=key, receipt=str(receipt_file), receipt_sha256=digest(receipt_file), final_sha256=receipt['final_sha256']))
        pre_root=DATA/'camera_robust_pre_native_v22_20261021'
        pre_record=json.loads((pre_root/'receipts'/(identity+'.json')).read_text())
        assert pre_record['source_sha256']==row['sha256'] and pre_record['outside_support_unchanged']
        assert digest(pre_root/pre_record['image'])==pre_record['image_sha256']
        output['images']['pre']=file_url(pre_root/pre_record['image'])
        rows.append(output)
        print('verified '+identity, flush=True)
    manifest = dict(models=models(), rows=rows, preprocessing='V2_2', development=bool(args.limit))
    text = render_html(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    (args.output.parent/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = dict(status='development' if args.limit else 'passed', images=len(rows), models=len(bindings), bindings=bindings, records=records, reused_acceptance='evaluation/vosr_six_model_evaluation_20261022/INDEPENDENT_AUDIT.json', html_sha256=digest(args.output))
    (args.output.parent/'COMPARISON_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(dict(images=len(rows), models=len(bindings), html=str(args.output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT/'index.html')
    parser.add_argument('--limit', type=int, default=0)
    build(parser.parse_args())
