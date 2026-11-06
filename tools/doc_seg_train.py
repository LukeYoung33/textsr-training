"""Supervised YOLO segmentation training with durable, truthful runtime status."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback


def atomic_json(path, value):
    """@luna-comment
    Input: Destination Path and JSON-compatible value.
    Output: Atomically replaced UTF-8 JSON file.
    Key Steps: Write temporary sibling then replace.
    Key Params: allow_nan=False rejects invalid telemetry.
    State/Dependencies: Filesystem; caller serializes concurrent writers.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def process_start(pid):
    """@luna-comment
    Input: Process integer ID.
    Output: Linux start-tick identity or None on unsupported platforms.
    Key Steps: Read proc stat after command-name closing parenthesis.
    Key Params: Field 22 identifies process creation.
    State/Dependencies: Linux proc filesystem when present.
    """
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class RunState:
    """Thread-safe atomic training status and independent heartbeat."""

    def __init__(self, directory, configuration):
        """@luna-comment
        Input: Run directory and CLI configuration mapping.
        Output: Initialized setup state.
        Key Steps: Capture process identity and start daemon heartbeat.
        Key Params: Heartbeat interval 30 seconds.
        State/Dependencies: Thread lock, event and status.json.
        """
        self.path = directory / "status.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.value = dict(stage="setup", pid=os.getpid(), process_start=process_start(os.getpid()),
                          started_at=time.time(), epoch=0, config=configuration,
                          training_complete=False, delivery_verified=False)
        self.update()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def update(self, **changes):
        """@luna-comment
        Input: Changed JSON status fields.
        Output: Durable snapshot with heartbeat and update timestamps.
        Key Steps: Lock, merge fields and atomically publish.
        Key Params: stage identifies actual lifecycle.
        State/Dependencies: Shared value and filesystem.
        """
        with self.lock:
            self.value.update(changes)
            self.value["heartbeat_at"] = time.time()
            if changes:
                self.value["updated_at"] = time.time()
            atomic_json(self.path, self.value)

    def heartbeat(self):
        """@luna-comment
        Input: Instance stop event.
        Output: Updated heartbeat while wrapper remains active.
        Key Steps: Wait interruptibly and publish.
        Key Params: Thirty-second period.
        State/Dependencies: RunState lock and daemon lifetime.
        """
        while not self.stop.wait(30):
            self.update()


def finite_metrics(mapping):
    """@luna-comment
    Input: Ultralytics scalar metric dictionary.
    Output: Finite float dictionary with unsupported values omitted.
    Key Steps: Convert numeric values and reject nonfinite entries.
    Key Params: None.
    State/Dependencies: No external state.
    """
    result = {}
    for key, value in mapping.items():
        try:
            number = float(value)
            if math.isfinite(number):
                result[str(key)] = number
        except (ValueError, TypeError):
            continue
    return result


def main():
    """@luna-comment
    Input: Training CLI, existing dataset YAML and checkpoint/config model.
    Output: Run artifacts and status; nonzero exit on failure.
    Key Steps: Train via public callbacks, reload best, independently validate.
    Key Params: Epoch budget, batch, resolution, seed and patience; no mosaic/flips.
    State/Dependencies: Isolated torch/Ultralytics environment and dataset.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    for name, default in (("epochs", 100), ("imgsz", 640), ("batch", 16),
                          ("workers", 4), ("patience", 25), ("seed", 42)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    directory = Path(args.project).resolve() / args.name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "status.json").exists() and not args.resume:
        parser.error("Run status already exists; choose a new name or explicit --resume.")
    if args.resume and not (directory / "weights/last.pt").is_file():
        parser.error("--resume requires weights/last.pt.")
    if args.resume and (directory / "status.json").exists():
        from doc_seg_monitor import alive
        previous = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if alive(previous):
            parser.error("Refusing resume while previous training process is alive.")
    state = RunState(directory, vars(args))
    try:
        from ultralytics import YOLO, __version__, settings
        import torch
        disabled = {key: False for key in ("sync", "hub", "clearml", "comet", "dvc", "mlflow",
                                           "neptune", "raytune", "tensorboard", "wandb")
                    if key in settings}
        settings.update(disabled)
        state.update(ultralytics_version=__version__, torch_version=torch.__version__,
                     disabled_integrations=disabled,
                     model_initialization="architecture_random" if not args.resume and
                     Path(args.model).suffix in (".yaml", ".yml") else "local_checkpoint")
        model = YOLO(str(directory / "weights/last.pt") if args.resume else args.model)

        def epoch_started(trainer):
            """@luna-comment
            Input: Public trainer callback object.
            Output: Running stage and current epoch snapshot.
            Key Steps: Read public epoch and save directory.
            Key Params: Epoch number is one-based for display.
            State/Dependencies: Parent RunState.
            """
            state.update(stage="running", current_epoch=int(trainer.epoch) + 1,
                         save_dir=str(trainer.save_dir), epoch_started_at=time.time())

        def epoch_finished(trainer):
            """@luna-comment
            Input: Public fit-epoch callback.
            Output: Completed epoch and finite validation metrics.
            Key Steps: Capture epoch after training and validation.
            Key Params: Epoch count denotes completed epochs.
            State/Dependencies: Parent RunState.
            """
            state.update(stage="running", epoch=int(trainer.epoch) + 1,
                         metrics=finite_metrics(getattr(trainer, "metrics", {})))

        def train_finished(trainer):
            """@luna-comment
            Input: Train-end callback.
            Output: Post-training pending-verification stage.
            Key Steps: Mark callback receipt without claiming delivery.
            Key Params: None.
            State/Dependencies: Parent RunState.
            """
            state.update(stage="validating", train_end_callback=True)

        model.add_callback("on_train_epoch_start", epoch_started)
        model.add_callback("on_fit_epoch_end", epoch_finished)
        model.add_callback("on_train_end", train_finished)
        parameters = dict(data=args.data, project=str(Path(args.project).resolve()), name=args.name,
                          epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                          workers=args.workers, patience=args.patience, seed=args.seed,
                          mosaic=0.0, fliplr=0.0, flipud=0.0, overlap_mask=False, exist_ok=True,
                          save=True, plots=True, deterministic=True, amp=args.amp)
        if args.resume:
            parameters["resume"] = True
        state.update(stage="running", effective_train_parameters=parameters)
        model.train(**parameters)
        best = directory / "weights/best.pt"
        if not best.is_file() or best.stat().st_size == 0:
            raise RuntimeError("Training returned without a nonempty best.pt")
        state.update(stage="validating", best_checkpoint=str(best))
        validated = YOLO(str(best))
        raw_metrics = validated.val(data=args.data, split="val", imgsz=args.imgsz,
                                             batch=args.batch, device=args.device, workers=args.workers,
                                             project=str(directory), name="final_validation",
                                             exist_ok=True).results_dict
        metrics = finite_metrics(raw_metrics)
        if not metrics or not any("mAP" in key for key in metrics) or any(
                "mAP" in key and key not in metrics for key in raw_metrics):
            raise RuntimeError("Final validation produced no finite mAP metrics")
        digest = hashlib.sha256(best.read_bytes()).hexdigest()
        atomic_json(directory / "validation.json", metrics)
        state.update(stage="training_complete", training_complete=True, delivery_verified=False,
                     finished_at=time.time(), validation_metrics=metrics, best_sha256=digest)
    except BaseException as error:
        state.update(stage="failed", error=str(error), traceback=traceback.format_exc(), finished_at=time.time())
        raise
    finally:
        state.stop.set()
        state.thread.join(timeout=5)


if __name__ == "__main__":
    main()
