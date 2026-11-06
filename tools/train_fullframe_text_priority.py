"""Extend the September 28 BasicSR training loop without editing legacy modules."""

from __future__ import annotations

import argparse
import sys
import types
import importlib
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def calculate_image_metric(data, opt):
    """@luna-comment
    Input: image metric inputs and explicit registered metric specification.
    Output: selected metric value with unchanged BasicSR registry semantics.
    Key Steps: copy options and call the registered image metric.
    Key Params: type chooses PSNR/SSIM or an explicitly imported image metric.
    State/Dependencies: process-local metric namespace; no global OCR import.
    """
    from basicsr.utils.registry import METRIC_REGISTRY
    settings = deepcopy(opt)
    metric_type = settings.pop("type")
    return METRIC_REGISTRY.get(metric_type)(**data, **settings)


def bootstrap_training_imports(*, enable_cp=False):
    """@luna-comment
    Input: new standalone entry process before legacy package registry discovery.
    Output: original BasicSR builders plus explicitly registered full-frame components.
    Key Steps: bypass unsafe top-level scans, retain original builder definitions,
        import only needed modules, and restore the discovery function afterward.
    Key Params: scan allowlist is confined to this process; enable_cp imports only
        the requested frozen feature-loss module and never global recognition metrics.
    State/Dependencies: local sys.modules and temporary basicsr.utils.scandir binding;
        source files and old entry processes remain unchanged.
    """
    packages = ("basicsr", "realesrgan", "basicsr.metrics", "realesrgan.data",
                "realesrgan.archs", "realesrgan.losses", "realesrgan.models", "realesrgan.metrics")
    for name in packages:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
            module.__package__ = name
            sys.modules[name] = module
    import basicsr.utils as utilities
    metrics = sys.modules["basicsr.metrics"]
    metrics.calculate_metric = calculate_image_metric
    metric_module = importlib.import_module("basicsr.metrics.psnr_ssim")
    metrics.calculate_psnr = metric_module.calculate_psnr
    metrics.calculate_ssim = metric_module.calculate_ssim
    original_scandir = utilities.scandir
    allowed = {
        str((ROOT / "basicsr/archs").resolve()).lower(): set(),
        str((ROOT / "basicsr/data").resolve()).lower(): set(),
        str((ROOT / "basicsr/models").resolve()).lower(): {"sr_model.py"},
        str((ROOT / "basicsr/losses").resolve()).lower(): {"basic_loss.py", "gan_loss.py"},
    }

    def selected_scandir(directory, *args, **kwargs):
        """@luna-comment
        Input: package scan directory and unchanged BasicSR scandir arguments.
        Output: original iterator restricted only for four known registry folders.
        Key Steps: compare resolved folder; filter filenames or delegate unchanged.
        Key Params: allowlist selects only required legacy modules in the new process.
        State/Dependencies: enclosing original scanner and immutable path allowlist.
        """
        choices = allowed.get(str(Path(directory).resolve()).lower())
        entries = original_scandir(directory, *args, **kwargs)
        return entries if choices is None else (entry for entry in entries if Path(entry).name in choices)

    try:
        utilities.scandir = selected_scandir
        for package in ("basicsr.archs", "basicsr.losses", "basicsr.data", "basicsr.models"):
            importlib.import_module(package)
    finally:
        utilities.scandir = original_scandir
    for module in (
        "realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch",
        "realesrgan.data.fullframe_text_priority_dataset",
        "realesrgan.losses.fullframe_text_priority_loss",
        "realesrgan.models.fullframe_text_priority_model",
    ):
        importlib.import_module(module)
    if enable_cp:
        importlib.import_module("realesrgan.losses.crnn_content_perceptual_loss")


def load_resume_state(opt):
    """@luna-comment
    Input: parsed BasicSR options for the new full-frame model, with resolved paths.
    Output: CPU-loaded optimizer/scheduler state or None; updates resume/checkpoint paths.
    Key Steps: find the newest numeric state in the configured directory, load,
        validate required fields, and bind the matching generator checkpoint.
    Key Params: auto_resume selects discovery; resume_state selects an exact state.
    State/Dependencies: torch and BasicSR check_resume; no legacy source mutation.
    """
    import torch
    from basicsr.utils import check_resume

    selected = opt["path"].get("resume_state")
    if opt.get("auto_resume"):
        directory = Path(opt["path"]["training_states"])
        candidates = []
        if directory.is_dir():
            for path in directory.glob("*.state"):
                if path.stem.isdecimal():
                    candidates.append((int(path.stem), path))
        if candidates:
            selected = str(max(candidates)[1])
    if not selected:
        return None
    path = Path(selected)
    if not path.is_file():
        raise FileNotFoundError(f"Resume state does not exist: {path}")
    # States are generated locally by BasicSR, and contain optimizer/scheduler data.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not {"epoch", "iter", "optimizers", "schedulers"} <= state.keys():
        raise ValueError("Malformed BasicSR training state")
    opt["path"]["resume_state"] = str(path)
    check_resume(opt, state["iter"])
    checkpoint = Path(opt["path"]["pretrain_network_g"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Matching generator checkpoint does not exist: {checkpoint}")
    return state


def validate_fullframe_options(options):
    """@luna-comment
    Input: effective new model options including BasicSR command-line overrides.
    Output: validated options or explicit errors before any training artifacts are made.
    Key Steps: reject unimplemented preprocessing/degradation and check train/val
        physical assets and independent source/capture IDs for leakage.
    Key Params: manifest paths resolve against their explicit dataset root.
    State/Dependencies: read-only configuration/asset validation, no legacy mutation.
    """
    from tools.make_fullframe_text_priority_configs import read_manifest
    if options.get("model_type") != "FullFrameTextPriorityModel":
        raise ValueError("The new entry cannot execute a legacy model")
    if options.get("pipeline_template", False):
        raise ValueError("Generate a concrete configuration from full-frame manifests before training")
    preprocessing = options.get("lq_preprocess")
    if preprocessing and (not isinstance(preprocessing, dict) or preprocessing.get("enabled", False)):
        raise ValueError("Full-frame pipeline currently requires disabled lq_preprocess")
    if options.get("high_order_degradation") or options.get("gt_usm") or options.get("queue_size", 0):
        raise ValueError("Full-frame pipeline requires paired native LQ/GT without online degradation/USM/queue")
    train_ids, val_ids = set(), set()
    for phase, settings in options.get("datasets", {}).items():
        if settings.get("type") != "FullFrameTextPriorityDataset":
            raise ValueError("New entry requires FullFrameTextPriorityDataset for every split")
        root = Path(settings.get("root", "."))
        manifest = Path(settings["manifest"])
        manifest = manifest if manifest.is_absolute() else root / manifest
        _, identities = read_manifest(manifest, root)
        (train_ids if phase == "train" else val_ids).update(identities)
    if not train_ids:
        raise ValueError("Training dataset is required")
    if train_ids & val_ids:
        raise ValueError(f"Train/validation source leakage: {sorted(train_ids & val_ids)}")


def make_epoch_sampler(original_sampler, runtime):
    """@luna-comment
    Input: unchanged BasicSR sampler class and process-local parsed resume context.
    Output: extension sampler class refreshing dataset epochs and skipping consumed batches.
    Key Steps: inherit original rank/shuffle logic; reconstruct the dataset epoch;
        calculate saved batch offset and exclude only previously consumed rank indices.
    Key Params: batch size and enlargement are fixed by the resumed configuration.
    State/Dependencies: new process only; persistent workers are rejected by dataset.
    """
    class FullFrameEpochSampler(original_sampler):
        def __init__(self, dataset, num_replicas, rank, ratio=1):
            """@luna-comment
            Input: new dataset, rank topology and unchanged BasicSR enlargement ratio.
            Output: original sampler plus reproducible resume epoch/index offset.
            Key Steps: initialize old sampler; validate batch budget; derive saved offset.
            Key Params: rank-local batch size defines consumed index count.
            State/Dependencies: runtime resume dictionary may advance a completed epoch.
            """
            super().__init__(dataset, num_replicas, rank, ratio)
            self.resume_epoch = None
            self.skip_samples = 0
            state = runtime.get("state")
            if state is None:
                return
            size = int(runtime["options"]["datasets"]["train"]["batch_size_per_gpu"])
            batches = self.num_samples // size
            if batches <= 0:
                raise ValueError("Training sampler cannot produce a complete batch")
            consumed = int(state["iter"]) - int(state["epoch"]) * batches
            if not 0 <= consumed <= batches:
                raise ValueError("Resume epoch/iteration does not match the configured data and batch budget")
            if consumed == batches:
                state["epoch"] = int(state["epoch"]) + 1
                consumed = 0
            self.resume_epoch = int(state["epoch"])
            self.skip_samples = consumed * size

        def set_epoch(self, epoch):
            """@luna-comment
            Input: epoch selected by the existing train loop. Output: refreshed tile plan.
            Key Steps: update dataset before worker creation, then original sampler epoch.
            Key Params: dataset seed+epoch reconstructs identical plans after resume.
            State/Dependencies: worker persistence is disabled; no old dataset affected.
            """
            self.dataset.set_epoch(epoch)
            super().set_epoch(epoch)

        def __iter__(self):
            """@luna-comment
            Input: current epoch sampler state. Output: rank indices excluding saved prefix.
            Key Steps: use unchanged old shuffle/rank distribution and slice resume offset.
            Key Params: skip applies only to the first resumed epoch.
            State/Dependencies: no RNG or dataset state is mutated here.
            """
            indices = list(super().__iter__())
            offset = self.skip_samples if self.epoch == self.resume_epoch else 0
            return iter(indices[offset:])
    return FullFrameEpochSampler


def main(argv=None):
    """@luna-comment
    Input: -opt YAML and the unchanged BasicSR launcher/debug/force_yml arguments.
    Output: training artifacts from the existing loop and a process exit status.
    Key Steps: restrict the entry to the new model, import registrations, install
        a process-local resume adapter, delegate, and restore process globals.
    Key Params: num_gpu=0 enables CPU; all BasicSR training flags retain semantics.
    State/Dependencies: modifies sys.argv and basicsr.train.load_resume_state only
        during this new entry; restores both even when training fails.
    """
    import yaml

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-opt", required=True)
    parser.add_argument("--force_yml", nargs="+")
    if "--help" in arguments or "-h" in arguments:
        print("Usage: python tools/train_fullframe_text_priority.py -opt CONFIG "
              "[--auto_resume] [--launcher none|pytorch|slurm] [--force_yml KEY=VALUE ...]")
        return 0
    parsed, _ = parser.parse_known_args(arguments)
    with Path(parsed.opt).open(encoding="utf-8-sig") as stream:
        options = yaml.safe_load(stream)
    if not isinstance(options, dict) or options.get("model_type") != "FullFrameTextPriorityModel":
        raise ValueError("This entry accepts only FullFrameTextPriorityModel; use the original entry for old runs")
    bootstrap_training_imports()
    if parsed.force_yml:
        from basicsr.utils.options import _postprocess_yml_value
        for expression in parsed.force_yml:
            keys, value = expression.split("=", 1)
            destination = options
            parts = keys.strip().split(":")
            for key in parts[:-1]:
                destination = destination[key]
            destination[parts[-1]] = _postprocess_yml_value(value.strip())
    validate_fullframe_options(options)
    if options.get("train", {}).get("fullframe_loss_opt", {}).get("cp_weight", 0) > 0:
        importlib.import_module("realesrgan.losses.crnn_content_perceptual_loss")
    import basicsr.train as legacy_train

    original_loader = legacy_train.load_resume_state
    original_sampler = legacy_train.EnlargedSampler
    original_argv = sys.argv
    runtime = {}

    def resume_adapter(opt):
        """@luna-comment
        Input: final parsed options. Output: CPU resume state and sampler runtime context.
        Key Steps: load state, retain its shared dictionary, and pass options to sampler.
        Key Params: unchanged train loop controls subsequent model and optimizer restore.
        State/Dependencies: enclosing runtime context only; old source remains unchanged.
        """
        state = load_resume_state(opt)
        runtime.update(options=opt, state=state)
        return state
    try:
        legacy_train.load_resume_state = resume_adapter
        legacy_train.EnlargedSampler = make_epoch_sampler(original_sampler, runtime)
        sys.argv = [str(Path(__file__)), *arguments]
        legacy_train.train_pipeline(str(ROOT))
    finally:
        legacy_train.load_resume_state = original_loader
        legacy_train.EnlargedSampler = original_sampler
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
elif __name__ == "__mp_main__":
    # Windows DataLoader workers execute this module before unpickling datasets.
    # Register the same minimal namespace without initializing global OCR assets.
    bootstrap_training_imports()
