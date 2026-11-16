"""部署近似结构；保留原训练参数名称，仅切换 GELU 的近似形式。"""
from torch import nn
from idea010_repvit_arch import RepViTM06NativeRestore


class RepViTM06TanhNativeRestore(RepViTM06NativeRestore):
    def __init__(self):
        """@luna-comment Input:无；Output:tanh-GELU恢复结构。
        Key Steps:构造原结构后切换18处GELU；Key Params:approximate=tanh。
        State/Dependencies:参数键不变；稳定Tanh和SE GPU修复仍由转换阶段应用。
        """
        super().__init__()
        for module in self.modules():
            if isinstance(module, nn.GELU):
                module.approximate = 'tanh'
