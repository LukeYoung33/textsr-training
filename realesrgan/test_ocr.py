import logging
import sys
from os import path as osp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import torch
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

import realesrgan.archs
import realesrgan.data
import realesrgan.models


def build_test_dataloader(dataset, dataset_opt):
    num_workers = int(dataset_opt.get("num_worker_per_gpu", 0))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(dataset_opt.get("batch_size_per_gpu", 1)),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(dataset_opt.get("pin_memory", False)),
        "persistent_workers": bool(dataset_opt.get("persistent_workers", False)) and num_workers > 0,
    }
    if hasattr(dataset, "collate_fn") and callable(dataset.collate_fn):
        loader_kwargs["collate_fn"] = dataset.collate_fn
    return DataLoader(**loader_kwargs)


def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt["path"]["log"], f"test_ocr_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for _, dataset_opt in sorted(opt["datasets"].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_test_dataloader(test_set, dataset_opt)
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}; "
            f"batch_size={dataset_opt.get('batch_size_per_gpu', 1)}, "
            f"num_workers={dataset_opt.get('num_worker_per_gpu', 0)}"
        )
        test_loaders.append(test_loader)

    model = build_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]
        logger.info(f"Testing {test_set_name}...")
        model.validation(test_loader, current_iter=opt["name"], tb_logger=None, save_img=opt["val"]["save_img"])


if __name__ == "__main__":
    test_pipeline(ROOT_PATH)
