# flake8: noqa
import os.path as osp
import sys
sys.path.append('./')
if __name__ == '__main__':
    import yaml
    from realesrgan.lenovo_equivalence_gate import verify
    if '-opt' in sys.argv:
        _option_path = sys.argv[sys.argv.index('-opt') + 1]
        with open(_option_path, encoding='utf-8') as _option_file:
            verify(yaml.safe_load(_option_file), _option_path)
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.losses
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
