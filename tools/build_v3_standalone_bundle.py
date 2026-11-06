"""Build a source-independent V2.2 pipeline, verified V3 exports and original weights."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
HISTORY = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/v3_native_four_20261015')


def sha(path):
    """@luna-comment Input:file. Output:SHA256. Key Steps:stream bounded chunks. Key Params:4MiB. State/Dependencies:read-only source."""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):
            value.update(block)
    return value.hexdigest()


def generated_sources(package):
    """@luna-comment
    Input:staged exact source and package root. Output:standalone color and native render modules.
    Key Steps:AST-extract transitive color helpers; retain renderer bodies and change only module imports.
    Key Params:original Y4UV constants and signed comments. State/Dependencies:no training or BasicSR imports in generated runtime.
    """
    color_source = (ROOT/'color_source.py').read_text()
    nodes = {node.name:node for node in ast.parse(color_source).body if isinstance(node,ast.FunctionDef)}
    names = {'rgb_to_y4uv','y4uv_to_yuv444'}
    while True:
        more = names|{node.id for name in names for node in ast.walk(nodes[name]) if isinstance(node,ast.Name) and node.id in nodes}
        if more==names:break
        names = more
    content = 'from __future__ import annotations\nimport torch\nimport torch.nn.functional as F\nfrom typing import Tuple\nY4UV_CHANNELS=6\nY4_CHANNELS=4\n'
    content += '\n\n'.join(ast.get_source_segment(color_source,nodes[name]) for name in sorted(names))
    utility = ast.parse((ROOT/'utils_source.py').read_text())
    conversion = next(node for node in utility.body if isinstance(node,ast.FunctionDef) and node.name=='YUV2RGB')
    conversion.body[0] = ast.Expr(value=ast.Constant(value='@luna-comment\nInput:normalized YUV444 and options. Output:RGB/BGR tensor. Key Steps:fixed UV offset and exact original conversion matrix. Key Params:is_train controls clamp. State/Dependencies:torch, no input mutation.'))
    content += '\n\n'+ast.unparse(conversion)+'\n'
    (package/'color.py').write_text(content)
    selected = []
    for source,names in [('native_source.py',('positions','restore')),('six_source.py',('restore_block',))]:
        tree = ast.parse((ROOT/source).read_text())
        for node in tree.body:
            if not isinstance(node,ast.FunctionDef) or node.name not in names:continue
            if node.name=='restore':node.name='restore_region'
            for parent in ast.walk(node):
                if isinstance(parent,ast.ImportFrom) and parent.module in ('realesrgan.data.basic_text_line_y4uv_v3','realesrgan.utils'):
                    parent.module='color'
            selected.append(ast.unparse(node))
    (package/'native_restore.py').write_text('import cv2\nimport numpy as np\n\n'+'\n\n'.join(selected)+'\n')


def model_entries():
    """@luna-comment Input:approved training roots. Output:nine explicit final model sources/lineages. Key Steps:map original and VOSR groups. Key Params:all250K, no substitution. State/Dependencies:immutable configs/checkpoints."""
    result = []
    for arm,identity,route in [('01_0921data_degraded','history_0921_v3_synthetic','synthetic'),('02_displayfit_paired','displayfit_v3_real','real'),('03_displayfit_degraded','displayfit_v3_synthetic','synthetic'),('04_displayfit_mixed75','displayfit_v3_mixed75','mixed75')]:
        result.append((identity,HISTORY/'configs'/(arm+'.yml'),HISTORY/'runs'/arm/'models/net_g_250000.pth',dict(round='20261015',target='legacy_0921_GT' if arm.startswith('01') else 'original_aligned_GT',route=route,preprocessing='none'),'region96'))
    rounds = [('adaptive_preprocessed_augmented_v1_20261020','20261020',[('02_v3_real','original_gt_v3_real','real','LQ_only_frozen_V1','original_aligned_GT'),('04_v3_synthetic','original_gt_v3_synthetic','synthetic','LQ_only_frozen_V1','original_aligned_GT')]),('adaptive_vosr_preprocessed_augmented_v1_20261021','20261021',[('02_v3_preprocessed','vosr_v3_preprocessed','real','LQ_and_GT_frozen_V1','GT_VOSR'),('04_v3_raw','vosr_v3_raw','real','none','GT_VOSR')]),('vosr_fullimage_block256_v1_20261021','20261021',[('06_v3_block256_raw','vosr_v3_block256_raw','real','none','GT_VOSR')])]
    import yaml
    for folder,date,arms in rounds:
        run = DATA/folder/('training_comparison_'+date)
        for arm,identity,route,tone,target in arms:
            config = run/'configs'/(arm+'.yml')
            options = yaml.safe_load(config.read_text())
            result.append((identity,config,run/'experiments'/options['name']/'models/net_g_250000.pth',dict(round=date,target=target,route=route,preprocessing=tone),'block256' if 'block256' in arm else 'region96'))
    return result


def cached_parameters(module):
    """@luna-comment Input:export-only ECB module. Output:registered fused kernel/bias. Key Steps:return precomputed source rep_params values. Key Params:no runtime CPU tensor creation. State/Dependencies:export clone only."""
    return module._export_kernel,module._export_bias


def export_clone(network):
    """@luna-comment
    Input:strict-loaded eval network. Output:independent export clone with cached ECB eval parameters.
    Key Steps:deep-copy; compute each original rep_params once; register buffers and bind cache accessor.
    Key Params:original forward/activation/skip preserved exactly. State/Dependencies:no source/weight mutation; buffers relocate with device.
    """
    result = copy.deepcopy(network).eval()
    for module in result.modules():
        if hasattr(module,'rep_params'):
            kernel,bias = module.rep_params()
            module.register_buffer('_export_kernel',kernel.detach().clone())
            module.register_buffer('_export_bias',bias.detach().clone())
            module.rep_params = types.MethodType(cached_parameters,module)
    return result


def main():
    """@luna-comment
    Input:seed ZIP/output root. Output:verified standalone package and model registry.
    Key Steps:strict eager loads, TorchScript exports, multi-batch numerical parity, copy original weights/configs/assets.
    Key Params:CPU FP32 tolerance1e-6, batches1/3/8. State/Dependencies:known remote training environment; original sources untouched.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    package = args.output/'textsr_v22_v3'
    package.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.seed) as archive:archive.extractall(package)
    shutil.copyfile(ROOT/'worker_source.py',package/'worker.py')
    generated_sources(package)
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    import torch
    import yaml
    from basicsr.archs import build_network
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261022)
    registry = {}
    for identity,config,checkpoint,training,strategy in model_entries():
        options = yaml.safe_load(config.read_text())
        logs = list(checkpoint.parent.parent.glob('train_*.log'))
        if not logs or not any('End of training' in path.read_text(errors='replace') for path in logs):
            raise ValueError('Missing completion proof '+identity)
        network = build_network(options['network_g']).cpu().eval()
        network.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
        canvas = options['network_g']['canvas_hw']
        height,width = canvas[0]//2,canvas[1]//2
        example = torch.rand(1,6,height,width)
        exported = torch.jit.freeze(torch.jit.trace(export_clone(network),example,check_trace=True))
        destination = package/'models/v3'/identity
        destination.mkdir(parents=True,exist_ok=True)
        script = destination/'model.torchscript.pt'
        exported.save(str(script))
        loaded = torch.jit.load(str(script)).eval()
        errors = []
        with torch.inference_mode(),torch.jit.optimized_execution(False):
            for batch in (1,3,8):
                for kind in ('uniform','normal'):
                    value = torch.rand(batch,6,height,width) if kind=='uniform' else torch.randn(batch,6,height,width)
                    expected,actual = network(value),loaded(value)
                    error = float((expected-actual).abs().max())
                    if actual.shape!=expected.shape or not torch.isfinite(actual).all() or error>1e-6:
                        raise ValueError(f'TorchScript parity failed {identity}: batch={batch},kind={kind},max_abs={error},actual={tuple(actual.shape)},expected={tuple(expected.shape)}')
                    errors.append(dict(batch=batch,input=kind,max_abs=error))
        network.cuda().eval()
        deployed = torch.jit.load(str(script),map_location='cuda:0').eval()
        gpu_errors = []
        with torch.inference_mode(),torch.jit.optimized_execution(False):
            for batch in (1,3,8):
                value = torch.rand(batch,6,height,width,device='cuda:0')
                difference = (network(value)-deployed(value)).abs()
                maximum,mean = float(difference.max()),float(difference.mean())
                if not torch.isfinite(difference).all() or maximum>1e-4 or mean>1e-6:
                    raise ValueError(f'CUDA export parity failed {identity}: {maximum}, {mean}')
                gpu_errors.append(dict(batch=batch,max_abs=maximum,mean_abs=mean))
        del deployed,network
        shutil.copyfile(checkpoint,destination/'net_g_250000.pth')
        shutil.copyfile(config,destination/'training.yml')
        metadata = dict(torchscript=str(script.relative_to(package)),checkpoint=str((destination/'net_g_250000.pth').relative_to(package)),checkpoint_sha256=sha(checkpoint),torchscript_sha256=sha(script),config_sha256=sha(config),source_checkpoint=str(checkpoint),source_config=str(config),iteration=250000,canvas_hw=canvas,strategy=strategy,training=training,export_parity=errors,cuda_parity=gpu_errors)
        (destination/'metadata.json').write_text(json.dumps(metadata,indent=2))
        registry[identity] = metadata
        print(identity,'export parity passed',flush=True)
    (package/'models/registry.json').write_text(json.dumps(registry,indent=2))
    ocr = Path('/home/guang.yang/workspace/PaddleOCR-3.7.0/deploy/official_models/PP-OCRv6_medium_det')
    shutil.copytree(ocr,package/'models/ocr/PP-OCRv6_medium_det',dirs_exist_ok=True)
    original = DATA/'adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020/evaluation/camera_final_20261021/source'
    (package/'examples').mkdir(exist_ok=True)
    for identity in ('013','045','169'):shutil.copyfile(original/(identity+'.jpg'),package/'examples'/(identity+'.jpg'))
    # Compare generated color/renderer interfaces directly against original source functions.
    spec = importlib.util.spec_from_file_location('portable_color',package/'color.py')
    portable = importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv,y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    rgb = torch.rand(3,3,96,480)
    packed = rgb_to_y4uv(rgb)
    if not torch.equal(packed,portable.rgb_to_y4uv(rgb)) or not torch.equal(y4uv_to_yuv444(packed),portable.y4uv_to_yuv444(packed)) or not torch.equal(YUV2RGB(y4uv_to_yuv444(packed),is_train=True),portable.YUV2RGB(portable.y4uv_to_yuv444(packed),is_train=True)):
        raise ValueError('Portable color conversion parity failed')
    (package/'EXPORT_VERIFIED.json').write_text(json.dumps(dict(status='passed',models=len(registry),color_conversion='byte_exact_tensor',batch_tests=[1,3,8],fp32_tolerance=1e-6),indent=2))
    for path in package.rglob('*.py'):
        code = path.read_text()
        if any(token in code for token in ('/AI_DNS/','/home/guang.yang/','E:/workspace/')):
            raise ValueError('External runtime path '+str(path))
    manifest = {str(path.relative_to(package)):sha(path) for path in sorted(package.rglob('*')) if path.is_file() and path.name!='ASSET_MANIFEST.json' and '__pycache__' not in path.parts}
    (package/'ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('Package ready',package,flush=True)


if __name__=='__main__':
    main()
