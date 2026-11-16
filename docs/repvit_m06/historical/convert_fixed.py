"""独立转换：仅修复已实证的SE广播及Tanh溢出，不加最终PACK。"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import subprocess
import os

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'repvit_m06_gelu_tanh_20261112'
SKILL=Path('E:/tools/skill-repo-master/skills/onnx2tflite-benchmark')


def main():
    """@luna-comment Input:已校验tanh ONNX；Output:FP32/FP16修复图；Key Steps:预检/私有lowering/原转换器/native-batch；Key Params:7SE/18Tanh；State/Dependencies:不改工具源码及生产接口。"""
    for name in ['RepViT_M06_R1ori_500K_b16_gelu_tanh.onnx','real_input.bin','reference_real.bin','EXPORT_VERIFIED.json','APPROXIMATION.json']:
        shutil.copyfile(BASE/name,HERE/name)
    for label in ['zeros','ones','random','real_lq16']:
        for suffix in ['input','reference','exact_reference']:
            shutil.copyfile(BASE/(label+'_'+suffix+'.npy'),HERE/(label+'_'+suffix+'.npy'))
    sys.path.insert(0,str(SKILL/'scripts'))
    from common import load_config
    sys.path.insert(0,load_config()['onnx2tflite_path'])
    from precheck_fix import precheck_and_fix
    source=HERE/'RepViT_M06_R1ori_500K_b16_gelu_tanh.onnx'
    fixed=precheck_and_fix(str(source),str(HERE/'converter.onnx'))
    import converter
    from gpu_batch_compat import conversion_overrides, spatial_gate_multiply
    from utils.op_registry import OPERATOR
    original_mul=OPERATOR.get('Mul')
    calls=[]

    class FixedMul(original_mul):
        def __call__(self,*args,**kwargs):
            """@luna-comment Input:原转换器乘法；Output:同语义同形SE乘法；Key Steps:命中静态gate形状后调用既有物化；Key Params:不拆卷积batch；State/Dependencies:仅本进程registry。"""
            a,b=self.first_operand,self.second_operand
            for feature,gate in [(a,b),(b,a)]:
                shape,small=tuple(feature.shape),tuple(gate.shape)
                if len(shape)==len(small)==4 and all(x is not None for x in shape+small) and shape[0]>1 and shape[1]>1 and shape[2]>1 and small==(shape[0],1,1,shape[3]):
                    calls.append(dict(feature=list(shape),gate=list(small)))
                    break
            return spatial_gate_multiply(a,b)

    records={}
    for precision in ['fp32','fp16']:
        calls.clear();folder=HERE/precision;folder.mkdir(exist_ok=True)
        existing=list(folder.glob('*.tflite'))
        if existing:
            from audit_native_batch import audit
            batch=audit(str(existing[0]),16)
            assert batch['pass'] and batch['operations'].get('RESIZE_NEAREST_NEIGHBOR',0)==7
            (folder/'native_batch.json').write_text(json.dumps(batch,indent=2))
            blob=existing[0].read_bytes()
            records[precision]=dict(path=str(existing[0]),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),se_lowering_count=7,stable_tanh=True,final_pack_added=False,native_batch_passed=True,reused_after_metadata_serialization_error=True)
            (HERE/'CONVERSION.json').write_text(json.dumps(records,indent=2));print('REUSED',precision,flush=True)
            continue
        try:
            with conversion_overrides(batch_compat=False,stable_tanh=True):
                OPERATOR.operator_dict['Mul']=FixedMul
                result=converter.onnx_converter(onnx_model_path=fixed,need_simplify=False,output_path=str(folder),output_node_names=None,target_formats=['tflite'],native_groupconv=False,weight_quant=False,fp16_model=precision=='fp16',int8_model=False,int8_mean=None,int8_std=None,image_root=None)
        finally:
            OPERATOR.operator_dict['Mul']=original_mul
        assert len(calls)==7,calls
        from audit_native_batch import audit
        batch=audit(result['tflite'],16)
        assert batch['pass'],batch['failures']
        (folder/'native_batch.json').write_text(json.dumps(batch,indent=2))
        blob=Path(result['tflite']).read_bytes()
        records[precision]=dict(path=result['tflite'],sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),se_lowerings=list(calls),stable_tanh=True,final_pack_added=False,native_batch_passed=True,converter_error=float(result['tflite_error']) if result.get('tflite_error') is not None else None)
        (HERE/'CONVERSION.json').write_text(json.dumps(records,indent=2));print('COMPLETE',precision,records[precision],flush=True)


if __name__=='__main__':main()
