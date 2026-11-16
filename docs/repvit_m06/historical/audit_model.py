"""记录原始TFLite每个节点、常量精度及与训练模型的输出差异。"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

HERE = Path(__file__).resolve().parent


def main():
    """@luna-comment Input:FP32/FP16原始转换产物；Output:节点CSV/精度JSON；Key Steps:FlatBuffer解析后实际Invoke；Key Params:保留失败；State/Dependencies:不修改模型，不启用近似适配。"""
    names = {v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    result = {}
    for precision in ['fp32','fp16']:
        paths = list((HERE/precision).glob('*.tflite'))
        if not paths:
            result[precision] = {'status':'conversion_failed'}
            continue
        path = paths[0]
        blob = path.read_bytes()
        model = schema.Model.GetRootAsModel(blob,0)
        sub = model.Subgraphs(0)
        rows = []
        for index in range(sub.OperatorsLength()):
            op = sub.Operators(index)
            code = model.OperatorCodes(op.OpcodeIndex())
            name = names.get(code.BuiltinCode(),str(code.BuiltinCode()))
            custom = (code.CustomCode() or b'').decode()
            tensor_names = [sub.Tensors(int(t)).Name().decode() for t in op.OutputsAsNumpy() if t >= 0]
            rows.append(dict(node=index,op=custom or name,version=code.Version(),inputs=op.InputsAsNumpy().tolist(),outputs=op.OutputsAsNumpy().tolist(),output_names=tensor_names))
        with (HERE/(precision+'_nodes.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        record=dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),nodes=rows,operators=dict(Counter(r['op'] for r in rows)),float16_tensors=sum(sub.Tensors(t).Type()==schema.TensorType.FLOAT16 for t in range(sub.TensorsLength())))
        try:
            interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=2)
            interpreter.allocate_tensors()
            inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
            record['input_shape']=inp['shape'].tolist();record['output_shape']=out['shape'].tolist()
            assert inp['shape'].tolist()==out['shape'].tolist()==[16,128,128,6]
            assert inp['dtype']==out['dtype']==np.float32
            errors={}
            for label in ['zeros','ones','random','real_lq16']:
                value=np.load(HERE/(label+'_input.npy')).transpose(0,2,3,1).copy()
                ref=np.load(HERE/(label+'_reference.npy')).transpose(0,2,3,1)
                interpreter.set_tensor(inp['index'],value);interpreter.invoke()
                got=interpreter.get_tensor(out['index']);delta=np.abs(got-ref)
                errors[label]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),finite=bool(np.isfinite(got).all()))
            record['errors']=errors
            record['numeric_passed']=all(e['finite'] and e['max_abs'] < (1e-4 if precision=='fp32' else (.01 if label=='real_lq16' else .05)) for label,e in errors.items())
        except Exception as error:
            record['runtime_error']=repr(error)
        result[precision]=record
        (HERE/'TFLITE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(precision,record.get('operators'),record.get('errors'),record.get('runtime_error'),flush=True)


if __name__=='__main__':main()
