"""冻结训练权重，独立导出tanh-GELU近似副本并区分模型近似与转换误差。"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from collections import Counter
import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    """@luna-comment Input:冻结500K权重及真实b16输入；Output:tanh近似ONNX及分阶段误差；Key Steps:SHA/strict/fuse/GELU/export/ORT；Key Params:转换1e-4；State/Dependencies:仅修改副本18处GELU，SE不变。"""
    torch.set_num_threads(2)
    binding = json.loads((ROOT/'evaluation/vit_six_mojito32_20261112/MODEL_BINDINGS.json').read_text(encoding='utf-8'))['repvit_ori']
    hashes = json.loads((ROOT/'evaluation/vit_six_mojito32_20261112/ARCHITECTURE_BINDINGS.json').read_text())
    runtime = HERE/'runtime'
    runtime.mkdir(exist_ok=True)
    for name in ['idea009_evit_arch.py', 'idea010_repvit_arch.py']:
        source = Path('Y:/guang.yang/TextSR/data/idea010_repvit_m06_20261111/code')/name
        blob = source.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == hashes[name]
        (runtime/name).write_bytes(blob)
    sys.path.insert(0, str(runtime))
    from idea010_repvit_arch import RepViTM06NativeRestore, RepDW, fuse
    source = Path(binding['checkpoint'].replace('/AI_DNS/', 'Y:/'))
    blob = source.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == binding['checkpoint_sha256']
    (HERE/'checkpoint.pth').write_bytes(blob)
    cfg = Path(binding['config'].replace('/AI_DNS/', 'Y:/')).read_bytes()
    assert hashlib.sha256(cfg).hexdigest() == binding['config_sha256']
    (HERE/'training_config.yml').write_bytes(cfg)
    model = RepViTM06NativeRestore().eval()
    raw = torch.load(HERE/'checkpoint.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(raw['params'], strict=True)
    fused = fuse(model)
    gelu_names = []
    for name, module in fused.named_modules():
        if isinstance(module, torch.nn.GELU):
            module.approximate = 'tanh'
            gelu_names.append(name)
    assert len(gelu_names) == 18
    assert not any(isinstance(m, (RepDW, torch.nn.BatchNorm2d)) for m in fused.modules())
    shutil.copyfile(ROOT/'evaluation/v3m_gt_plus1_500k_tflite_20261110/real_input.bin', HERE/'real_input.bin')
    real = np.fromfile(HERE/'real_input.bin', dtype=np.float32).reshape(16,128,128,6).transpose(0,3,1,2).copy()
    rng = np.random.default_rng(20261112)
    tests = dict(zeros=np.zeros_like(real), ones=np.ones_like(real), random=rng.random(real.shape,dtype=np.float32), real_lq16=real)
    errors, refs = {}, {}
    for name, value in tests.items():
        with torch.no_grad():
            ref = model(torch.from_numpy(value)).numpy()
            got = fused(torch.from_numpy(value)).numpy()
        error = float(np.abs(ref-got).max())
        assert np.isfinite(got).all(), (name,error)
        errors[name] = error
        refs[name] = got
        np.save(HERE/(name+'_exact_reference.npy'),ref)
        np.save(HERE/(name+'_input.npy'),value)
        np.save(HERE/(name+'_reference.npy'),got)
    torch.save(fused.state_dict(),HERE/'fused.pth')
    path = HERE/'RepViT_M06_R1ori_500K_b16_gelu_tanh.onnx'
    torch.onnx.export(fused,torch.from_numpy(real),str(path),opset_version=16,input_names=['y4uv_input'],output_names=['y4uv_output'],dynamo=False,do_constant_folding=True)
    graph = onnx.shape_inference.infer_shapes(onnx.load(path))
    onnx.checker.check_model(graph)
    onnx.save(graph,path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    ort_errors = {}
    for name,value in tests.items():
        got = session.run(None,{'y4uv_input':value})[0]
        error = float(np.abs(got-refs[name]).max())
        assert np.isfinite(got).all() and error < 1e-4,(name,error)
        ort_errors[name] = error
    refs['real_lq16'].transpose(0,2,3,1).copy().tofile(HERE/'reference_real.bin')
    report = dict(source=binding,architecture_hashes=hashes,repdw_fused=sum(isinstance(m,RepDW) for m in model.modules()),bn_fused=sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),gelu_approximation_max_abs=errors,gelu_modules=gelu_names,onnx_errors=ort_errors,operators=dict(Counter(n.op_type for n in graph.graph.node)),onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),torch=torch.__version__,io_nchw=[16,6,128,128],other_adaptations=False)
    (HERE/'EXPORT_VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
