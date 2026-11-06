"""Paired, native-scale Lenovo GT-to-LQ diagnostic and stochastic pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ZOOM_BATCHES = {
    2: {"400-407", "407-520", "521-628"},
    3: {"133-177", "178-219", "220-235", "236-304", "305-333"},
    4: {"000-049", "050-132"},
}


def zoom_for(sample_id: str) -> int:
    token = sample_id.split("_", 1)[0]
    found = [zoom for zoom, batches in ZOOM_BATCHES.items() if token in batches]
    if len(found) != 1:
        raise ValueError(f"Unknown or ambiguous Zoom for {sample_id}")
    return found[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(manifest: Path):
    root = manifest.parent.parent
    result = []
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            file_map = row["files"]
            result.append({
                "id": row["sample_id"],
                "source": row["source_identity"],
                "zoom": zoom_for(row["sample_id"]),
                "gt": root / file_map["gt"],
                "lq": root / file_map["lq"],
                "text": root / file_map["text_mask"],
                "valid": root / file_map["valid_mask"],
                "ocr": root / file_map["ocr"],
            })
    return result


def audit(train: Path, val: Path, out: Path):
    train_rows, val_rows = records(train), records(val)
    train_sources = {row["source"] for row in train_rows}
    val_sources = {row["source"] for row in val_rows}
    if train_sources & val_sources:
        raise RuntimeError("Train/validation source identity overlap")
    counts = {}
    problems = []
    provenance = defaultdict(int)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        counts[name] = {str(z): sum(r["zoom"] == z for r in rows) for z in (2, 3, 4)}
        for row in rows:
            try:
                headers = [cv2.imread(str(row[k]), cv2.IMREAD_UNCHANGED) for k in ("gt", "lq", "text", "valid")]
                if any(image is None for image in headers):
                    raise ValueError("missing or unreadable image")
                if len({image.shape[:2] for image in headers}) != 1:
                    raise ValueError("different image dimensions")
                if any(image.shape[0] < 32 or image.shape[1] < 32 for image in headers):
                    raise ValueError("image too small")
                if not row["ocr"].is_file():
                    raise ValueError("missing OCR")
                if not np.any(headers[3]):
                    raise ValueError("empty valid mask")
                if not np.any(headers[2]):
                    raise ValueError("empty text mask")
                provenance[row["source"].split("/", 1)[0]] += 1
            except Exception as exc:
                problems.append({"split": name, "id": row["id"], "reason": str(exc)})
    report = {
        "train_manifest": str(train), "train_sha256": sha256(train),
        "val_manifest": str(val), "val_sha256": sha256(val),
        "counts": counts, "source_prefix_counts": dict(provenance),
        "source_overlap": len(train_sources & val_sources), "problems": problems,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": counts, "problems": len(problems), "report": str(out)}), flush=True)
    if problems:
        raise RuntimeError(f"{len(problems)} invalid image pairs; see {out}")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def crop_at(array: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    crop = array[y:y + size, x:x + size]
    pad_h, pad_w = size - crop.shape[0], size - crop.shape[1]
    if pad_h or pad_w:
        crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return crop


def load_crop(row, rng: random.Random, size: int = 128, fixed: bool = False):
    gt, lq = read_rgb(row["gt"]), read_rgb(row["lq"])
    text_mask = cv2.imread(str(row["text"]), cv2.IMREAD_GRAYSCALE)
    valid_mask = cv2.imread(str(row["valid"]), cv2.IMREAD_GRAYSCALE)
    if gt.shape != lq.shape or text_mask.shape != gt.shape[:2]:
        raise ValueError(f"Pair shape mismatch: {row['id']}")
    ys, xs = np.where((text_mask > 0) & (valid_mask > 0))
    if not len(xs):
        raise ValueError(f"No valid text: {row['id']}")
    point = len(xs) // 2 if fixed else rng.randrange(len(xs))
    x = max(0, min(gt.shape[1] - size, int(xs[point]) - size // 2))
    y = max(0, min(gt.shape[0] - size, int(ys[point]) - size // 2))
    arrays = [crop_at(a, x, y, size) for a in (gt, lq, text_mask, valid_mask)]
    valid = (arrays[3] > 0).astype(np.float32)
    valid[max(0, gt.shape[0] - y):, :] = 0
    valid[:, max(0, gt.shape[1] - x):] = 0
    return (
        torch.from_numpy(np.ascontiguousarray(arrays[0].transpose(2, 0, 1))),
        torch.from_numpy(np.ascontiguousarray(arrays[1].transpose(2, 0, 1))),
        torch.from_numpy((arrays[2] > 0).astype(np.float32)[None]),
        torch.from_numpy(valid[None]),
        (x, y),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(True),
                                    nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return F.relu(x + self.layers(x))


class DeterministicDegrader(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(4, 32, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(6)])
        self.tail = nn.Conv2d(32, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, gt, zoom, conditioned: bool):
        z = ((zoom.float() - 4) / 2 if conditioned else torch.zeros_like(zoom, dtype=torch.float32))
        z = z[:, None, None, None].expand(-1, 1, gt.shape[-2], gt.shape[-1])
        return gt + self.tail(self.blocks(F.relu(self.head(torch.cat((gt, z), 1)))))


def sobel(image):
    gray = image[:, :1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114
    kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    ky = kx.transpose(-2, -1)
    return F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)


def diagnostic_loss(pred, target, gt, text_mask, valid):
    gx, gy = sobel(gt)
    edge = ((gx.square() + gy.square()).sqrt() > 0.05).float() * text_mask
    edge = F.max_pool2d(edge, 5, stride=1, padding=2)
    inner = F.pad(valid[..., 1:-1, 1:-1], (1, 1, 1, 1))
    weights = valid * (1 + text_mask + edge)
    pixel = ((pred - target).abs() * weights).sum() / (3 * weights.sum().clamp_min(1))
    px, py = sobel(pred)
    tx, ty = sobel(target)
    band = edge * inner
    grad = (((px - tx).abs() + (py - ty).abs()) * band).sum() / (2 * band.sum().clamp_min(1))
    return pixel + 0.2 * grad, pixel.detach(), grad.detach()


def batch_for(grouped, rng, batch_size, size=128):
    rows = [rng.choice(grouped[rng.choice((2, 3, 4))]) for _ in range(batch_size)]
    crops = [load_crop(row, rng, size) for row in rows]
    return [torch.stack([crop[i] for crop in crops]) for i in range(4)], torch.tensor([row["zoom"] for row in rows])


def build_cache(manifest: Path, out: Path, per_source: int, size: int, seed: int):
    """Decode full source images once; retain fixed native-resolution crop coordinates."""
    rows = records(manifest)
    out.mkdir(parents=True, exist_ok=True)
    total = len(rows) * per_source
    arrays = {
        key: np.lib.format.open_memmap(out / f"{key}.npy", mode="w+", dtype=np.uint8, shape=shape)
        for key, shape in {
            "gt": (total, size, size, 3), "lq": (total, size, size, 3),
            "text": (total, size, size), "valid": (total, size, size),
        }.items()
    }
    rng = random.Random(seed)
    meta = []
    for index, row in enumerate(rows):
        gt = (read_rgb(row["gt"]) * 255).round().astype(np.uint8)
        lq = (read_rgb(row["lq"]) * 255).round().astype(np.uint8)
        text_mask = cv2.imread(str(row["text"]), cv2.IMREAD_GRAYSCALE)
        valid_mask = cv2.imread(str(row["valid"]), cv2.IMREAD_GRAYSCALE)
        if gt.shape != lq.shape or text_mask.shape != gt.shape[:2] or valid_mask.shape != gt.shape[:2]:
            raise ValueError(f"Pair shape mismatch: {row['id']}")
        ys, xs = np.where((text_mask > 0) & (valid_mask > 0))
        if len(xs) == 0:
            raise ValueError(f"No valid text: {row['id']}")
        for repeat in range(per_source):
            point = rng.randrange(len(xs))
            x = max(0, min(gt.shape[1] - size, int(xs[point]) - size // 2))
            y = max(0, min(gt.shape[0] - size, int(ys[point]) - size // 2))
            position = index * per_source + repeat
            for key, source in (("gt", gt), ("lq", lq), ("text", text_mask), ("valid", valid_mask)):
                arrays[key][position] = crop_at(source, x, y, size)
            if gt.shape[0] - y < size:
                arrays["valid"][position, gt.shape[0] - y:] = 0
            if gt.shape[1] - x < size:
                arrays["valid"][position, :, gt.shape[1] - x:] = 0
            meta.append({"id": row["id"], "source": row["source"], "zoom": row["zoom"], "crop_xy": [x, y]})
        if (index + 1) % 100 == 0:
            print(json.dumps({"cached_sources": index + 1, "total_sources": len(rows)}), flush=True)
    for array in arrays.values():
        array.flush()
    (out / "meta.json").write_text(json.dumps({"manifest_sha256": sha256(manifest), "size": size,
                                               "per_source": per_source, "patches": meta}), encoding="utf-8")
    print(json.dumps({"cache": str(out), "patches": total}), flush=True)


class PatchCache:
    def __init__(self, folder: Path, manifest: Path, size: int):
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if meta["manifest_sha256"] != sha256(manifest) or meta["size"] != size:
            raise ValueError("Patch cache does not match manifest or crop size")
        self.arrays = {key: np.load(folder / f"{key}.npy", mmap_mode="r")
                       for key in ("gt", "lq", "text", "valid")}
        self.groups = {zoom: [i for i, patch in enumerate(meta["patches"]) if patch["zoom"] == zoom]
                       for zoom in (2, 3, 4)}
        self.zooms = [patch["zoom"] for patch in meta["patches"]]

    def batch(self, rng, batch_size):
        indices = [rng.choice(self.groups[rng.choice((2, 3, 4))]) for _ in range(batch_size)]
        images = [torch.from_numpy(np.array(self.arrays[key][indices], copy=True)).float() / 255
                  for key in ("gt", "lq")]
        masks = [torch.from_numpy(np.array(self.arrays[key][indices], copy=True)).float().unsqueeze(1) / 255
                 for key in ("text", "valid")]
        images = [image.permute(0, 3, 1, 2) for image in images]
        labels = [self.zooms[index] for index in indices]
        return images + masks, torch.tensor(labels)


def save_preview(run: Path, model, val_rows, device, conditioned):
    model.eval()
    folder = run / "preview"
    folder.mkdir(exist_ok=True)
    summary = []
    with torch.no_grad():
        for zoom in (2, 3, 4):
            chosen = [row for row in val_rows if row["zoom"] == zoom][:3]
            for row in chosen:
                gt, lq, text_mask, valid, xy = load_crop(row, random.Random(0), fixed=True)
                pred = model(gt[None].to(device), torch.tensor([zoom], device=device), conditioned)[0].clamp(0, 1).cpu()
                parts = [gt, lq, pred]
                image = np.concatenate([(p.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for p in parts], axis=1)
                cv2.imwrite(str(folder / f"{row['id']}_z{zoom}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                summary.append({"id": row["id"], "zoom": zoom, "crop_xy": xy,
                                "rgb_mae": float(((pred - lq).abs() * valid).sum() / (3 * valid.sum().clamp_min(1)))})
    (run / "preview.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model.train()


def validate(model, val_crops, device, conditioned):
    model.eval()
    by_zoom = defaultdict(list)
    with torch.no_grad():
        for row, crop in val_crops:
            gt, lq, text_mask, valid = [item[None].to(device) for item in crop[:4]]
            zoom = torch.tensor([row["zoom"]], device=device)
            pred = model(gt, zoom, conditioned).clamp(0, 1)
            pred_x, pred_y = sobel(pred)
            lq_x, lq_y = sobel(lq)
            gt_x, gt_y = sobel(gt)
            band = F.max_pool2d((((gt_x.square() + gt_y.square()).sqrt() > 0.05).float() * text_mask),
                                5, stride=1, padding=2) * valid
            band[..., :1, :] = 0
            band[..., -1:, :] = 0
            band[..., :, :1] = 0
            band[..., :, -1:] = 0
            edge_mae = (((pred_x - lq_x).abs() + (pred_y - lq_y).abs()) * band).sum() / (2 * band.sum().clamp_min(1))
            identity_edge_mae = (((gt_x - lq_x).abs() + (gt_y - lq_y).abs()) * band).sum() / (2 * band.sum().clamp_min(1))
            text_valid = text_mask * valid
            bg_valid = (1 - text_mask) * valid
            text_mae = ((pred - lq).abs() * text_valid).sum() / (3 * text_valid.sum().clamp_min(1))
            bg_mae = ((pred - lq).abs() * bg_valid).sum() / (3 * bg_valid.sum().clamp_min(1))
            by_zoom[row["zoom"]].append((float(edge_mae), float(identity_edge_mae), float(text_mae), float(bg_mae)))
    model.train()
    return {str(z): {key: float(np.mean(np.array(values)[:, i]))
                     for i, key in enumerate(("edge_mae", "identity_edge_mae", "text_mae", "bg_mae"))}
            for z, values in by_zoom.items()}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    train_rows, val_rows = records(args.train_manifest), records(args.val_manifest)
    grouped = {zoom: [row for row in train_rows if row["zoom"] == zoom] for zoom in (2, 3, 4)}
    if args.limit:
        grouped = {zoom: rows[:args.limit] for zoom, rows in grouped.items()}
    cache = PatchCache(args.cache, args.train_manifest, args.crop) if args.cache else None
    if args.limit:
        val_rows = [row for zoom in (2, 3, 4) for row in val_rows if row["zoom"] == zoom][:12]
    val_crops = [(row, load_crop(row, random.Random(0), args.crop, fixed=True)) for row in val_rows]
    if any(not group for group in grouped.values()):
        raise ValueError("Missing a 0920 Zoom group")
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"arm": args.arm, "steps": args.steps, "batch": args.batch, "seed": args.seed,
              "train_manifest": str(args.train_manifest), "train_sha256": sha256(args.train_manifest),
              "val_manifest": str(args.val_manifest), "val_sha256": sha256(args.val_manifest),
              "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeterministicDegrader().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), **config}), flush=True)
    started = time.time()
    for step in range(1, args.steps + 1):
        images, zoom = cache.batch(rng, args.batch) if cache else batch_for(grouped, rng, args.batch, args.crop)
        gt, lq, text_mask, valid = [item.to(device) for item in images]
        zoom = zoom.to(device)
        pred = model(gt, zoom, args.arm == "d1")
        loss, pixel, edge = diagnostic_loss(pred, lq, gt, text_mask, valid)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss), "pixel": float(pixel),
                              "edge": float(edge), "seconds": round(time.time() - started, 1)}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "config": config}, args.out / f"step_{step:06d}.pt")
            save_preview(args.out, model, val_rows, device, args.arm == "d1")
            metrics = validate(model, val_crops, device, args.arm == "d1")
            with (args.out / "validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "by_zoom": metrics}) + "\n")
            print(json.dumps({"step": step, "validation": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "cache", "train"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("d0", "d1"), default="d0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.train_manifest, args.val_manifest, args.out)
    elif args.mode == "cache":
        build_cache(args.train_manifest, args.out, args.per_source, args.crop, args.seed)
    else:
        train(args)


if __name__ == "__main__":
    main()
