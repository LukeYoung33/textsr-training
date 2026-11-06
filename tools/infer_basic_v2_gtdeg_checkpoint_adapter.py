"""GT-deg checkpoint adapter using the V4 adapter's detector/rotation geometry.

Vertical detector ROIs are normalized to horizontal by the shared deployment
pipeline, restored as height-96 tiled rows, inverse-rotated, and composited.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
LEGACY_RUNTIME_SITE = Path("/home/guang.yang/mplus/lib/python3.10/site-packages")
if LEGACY_RUNTIME_SITE.is_dir() and str(LEGACY_RUNTIME_SITE) not in sys.path:
    # ViT_env owns the GT-deg model dependencies; this legacy runtime provides
    # only the detector packages absent from that environment.
    sys.path.append(str(LEGACY_RUNTIME_SITE))
# The historic deployment adapter uses pyclipper for DB contour expansion.
# Keep that adapter untouched: on minimal inference environments a compatible
# rounded Shapely buffer is injected only for this new GT-deg entry point.
try:
    import pyclipper  # noqa: F401
    UNCLIP_BACKEND = "pyclipper"
except ModuleNotFoundError:
    from shapely.geometry import Polygon

    class _ShapelyClipperOffset:
        def AddPath(self, path, *_args):
            self._path = path

        def Execute(self, delta):
            polygon = Polygon(self._path).buffer(float(delta), join_style=1)
            if polygon.is_empty or polygon.geom_type != "Polygon":
                return []
            return [[(int(round(x)), int(round(y))) for x, y in polygon.exterior.coords[:-1]]]

    sys.modules["pyclipper"] = SimpleNamespace(
        PyclipperOffset=_ShapelyClipperOffset, JT_ROUND=1, ET_CLOSEDPOLYGON=1)
    UNCLIP_BACKEND = "shapely_round_fallback"
from infer_basic_v2_deployment_pipeline import DetectorONNXRunner, RestoreGeometry, Y4UV_SHAPE, SKIP_SHAPE, run_bgr_adapter_frame
from tools.basic_v2_eval_network import build_eval_network


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def _legacy_y4uv_to_nv21(value: np.ndarray) -> np.ndarray:
    """Pack the shared tile's Y4UV planes into the GT-deg NV21 contract.

    The source tile is already in the same normalized Y/U/V convention used by
    the GT-deg dataset.  Keeping the planes native avoids a lossy BGR color
    conversion before the model call.
    """
    if tuple(value.shape) != Y4UV_SHAPE: raise ValueError(f"expected {Y4UV_SHAPE}, got {value.shape}")
    x = np.asarray(value, dtype=np.float32)[0]
    y = x[:4].transpose(1, 2, 0).reshape(48, 240, 2, 2).transpose(0, 2, 1, 3).reshape(96, 480)
    u, v = x[4], x[8]
    vu = np.empty((48, 480), dtype=np.float32)
    vu[:, 0::2], vu[:, 1::2] = v, u
    return np.concatenate((y, vu), axis=0)[None, None].astype(np.float32)


def _v2_to_compositor_yuv(value: np.ndarray) -> np.ndarray:
    """Convert the model's native normalized YUV444 directly for composition."""
    output = np.asarray(value, dtype=np.float32)
    if output.ndim != 4 or output.shape[1] != 3:
        raise ValueError(f"expected model YUV444 output [1,3,H,W], got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("model output contains NaN/Inf")
    if float(output.max()) <= 1.5:
        output = output * 255.0
    return np.clip(output, 0.0, 255.0).astype(np.float32, copy=False)


class GTDegRunner:
    def __init__(self, model: torch.nn.Module, device: torch.device): self.model, self.device, self.calls = model, device, 0
    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        if tuple(inputs["y4uv"].shape) != Y4UV_SHAPE or tuple(inputs["skip"].shape) != SKIP_SHAPE: raise ValueError("invalid shared tile")
        with torch.inference_mode(): out = self.model(torch.from_numpy(_legacy_y4uv_to_nv21(inputs["y4uv"])).to(self.device)).detach().cpu().numpy()
        self.calls += 1
        return _v2_to_compositor_yuv(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--detector", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--network-config", required=True); p.add_argument("--output", required=True); p.add_argument("--trace"); p.add_argument("--debug-dir"); p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); device = torch.device(a.device if a.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, digest = build_eval_network(Path(a.checkpoint), device, Path(a.network_config)); image = cv2.imread(a.input, cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(a.input)
    out_frame, output, trace = run_bgr_adapter_frame(image, DetectorONNXRunner(a.detector), GTDegRunner(model, device), debug_dir=a.debug_dir, restore_geometry=RestoreGeometry(content_h=96, overlap=64, interpolation=cv2.INTER_AREA, force_even_width=True))
    trace.update({"input": a.input, "output": a.output, "checkpoint_sha256": digest, "architecture": "BasicTextLineR4StudentV2 GT-deg", "vertical_policy": "rotate clockwise to horizontal before tile restore; inverse rotate before composition", "detector_unclip_backend": UNCLIP_BACKEND})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(a.output, output)
    Path(a.trace or (a.output + ".json")).write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status":"passed", "accepted_count":trace.get("accepted_count", 0), "output":a.output}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
