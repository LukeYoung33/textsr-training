"""Verify and package a trained document segmentation checkpoint."""
import argparse
from collections import defaultdict, deque
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from urllib.parse import quote


def write_json(path, payload):
    """@luna-comment
    Input: Destination and JSON-compatible payload.
    Output: Atomic UTF-8 snapshot.
    Key Steps: Write sibling temporary file, replace destination.
    Key Params: Nonfinite numbers are forbidden.
    State/Dependencies: Filesystem.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def select_samples(records, count):
    """@luna-comment
    Input: Prepared manifest records and positive sample limit.
    Output: Deterministic source-stratified records from test or fallback val.
    Key Steps: Prefer test globally, group by source, round-robin sorted paths.
    Key Params: count bounds copied examples.
    State/Dependencies: No mutable external state.
    """
    split = "test" if any(row["split"] == "test" for row in records) else "val"
    groups = defaultdict(list)
    for row in records:
        if row["split"] == split:
            groups[row["source"]].append(row)
    queues = [deque(sorted(group, key=lambda row: row["prepared_image"]))
              for _, group in sorted(groups.items())]
    selected = []
    while len(selected) < count and any(queues):
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
    if not selected:
        raise ValueError("Manifest contains no test or validation samples")
    return selected


def verify_predictions(directory, expected):
    """@luna-comment
    Input: Inference output directory and expected number of input images.
    Output: Geometry/artifact summary; raises on malformed or all-empty output.
    Key Steps: Verify complete run, decode image artifacts and check finite convex bounded quads.
    Key Params: At least one boundary/approximate quad; homography corner error at most 0.02 pixels.
    State/Dependencies: Inference JSON schema, OpenCV, NumPy and generated files.
    """
    import cv2
    import numpy as np
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "completed" or run.get("errors") or len(run.get("results", [])) != expected:
        raise ValueError("Inference run failed or did not process all expected samples")
    total, empty, rejected = 0, 0, 0
    geometry_types = defaultdict(int)
    maximum_corner_error = 0.0
    for item in run["results"]:
        path = Path(item["result_json"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        width, height = payload["image_width"], payload["image_height"]
        if width <= 0 or height <= 0:
            raise ValueError("Invalid source dimensions")
        overlay = path.parent / payload["overlay_file"]
        if not overlay.is_file() or overlay.stat().st_size == 0:
            raise ValueError(f"Missing overlay: {overlay}")
        decoded = cv2.imdecode(np.fromfile(overlay, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.size == 0 or decoded.shape[:2] != (height, width):
            raise ValueError(f"Invalid overlay pixels or dimensions: {overlay}")
        instances = payload["instances"]
        if item["instance_count"] != len(instances):
            raise ValueError("Per-image instance count mismatch")
        empty += not instances
        rejected += len(payload.get("rejected", []))
        for rejection in payload.get("rejected", []):
            if "warp" in rejection or "image_file" in rejection:
                raise ValueError("Rejected boundary must not have a warp artifact")
        for instance in instances:
            geometry_type = instance.get("geometry_type")
            if geometry_type not in ("boundary_quad", "approximate_quad"):
                raise ValueError("Only boundary_quad or approximate_quad counts as warp success; diagnostic crops are excluded")
            contour = np.asarray(instance.get("raw_contour_xy", []), dtype=np.float64)
            if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3 or not np.isfinite(contour).all():
                raise ValueError("Boundary output must retain a finite original contour")
            quad = instance["quad"]
            if len(quad) != 4 or any(len(point) != 2 for point in quad):
                raise ValueError("Output geometry must contain exactly four xy vertices")
            if not all(math.isfinite(float(v)) for point in quad for v in point):
                raise ValueError("Nonfinite quadrilateral")
            outside = not all(0 <= x <= width - 1 and 0 <= y <= height - 1 for x, y in quad)
            if outside and geometry_type != "approximate_quad":
                raise ValueError("Quadrilateral extends outside image")
            crosses = []
            for index in range(4):
                a, b, c = quad[index], quad[(index + 1) % 4], quad[(index + 2) % 4]
                crosses.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
            if not (all(v > 1e-6 for v in crosses) or all(v < -1e-6 for v in crosses)):
                raise ValueError("Nonconvex, degenerate or crossed quadrilateral")
            crop = path.parent / instance["image_file"]
            if not crop.is_file() or crop.stat().st_size == 0:
                raise ValueError(f"Missing crop: {crop}")
            decoded = cv2.imdecode(np.fromfile(crop, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.size == 0 or min(decoded.shape[:2]) < 1:
                raise ValueError(f"Invalid crop pixels: {crop}")
            warp = instance.get("warp", {})
            if geometry_type == "approximate_quad":
                expected_padding = "constant_white_not_reconstructed" if outside else "none"
                if (warp.get("padding_applied") is not outside or
                        warp.get("unknown_region_padding") != expected_padding or
                        warp.get("border_mode") != "constant_white"):
                    raise ValueError("Approximate quad must explicitly and correctly declare unknown-region white padding")
                extent_limit = warp.get("outside_extent_limit_pixels")
                if not isinstance(extent_limit, (int, float)) or not math.isfinite(extent_limit) or extent_limit <= 0:
                    raise ValueError("Missing finite out-of-frame extent limit")
                if not all(-extent_limit <= x <= width-1+extent_limit and
                           -extent_limit <= y <= height-1+extent_limit for x, y in quad):
                    raise ValueError("Approximate quad exceeds declared out-of-frame extent limit")
            transform = np.asarray(warp.get("image_to_output", []), dtype=np.float64)
            if warp.get("transform_type") != "homography" or transform.shape != (3, 3) or not np.isfinite(transform).all():
                raise ValueError("Boundary warp must retain its finite 3x3 homography")
            if np.linalg.matrix_rank(transform) != 3:
                raise ValueError("Singular warp homography")
            output_width, output_height = warp.get("output_width"), warp.get("output_height")
            if not isinstance(output_width, int) or not isinstance(output_height, int) or min(output_width, output_height) < 2:
                raise ValueError("Invalid warp output dimensions")
            if decoded.shape[:2] != (output_height, output_width):
                raise ValueError("Warp PNG size does not match recorded homography destination")
            homogeneous = np.column_stack((np.asarray(quad, dtype=np.float64), np.ones(4))) @ transform.T
            if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
                raise ValueError("Warp maps a corner to infinity")
            projected = homogeneous[:, :2] / homogeneous[:, 2:3]
            target = np.asarray([[0, 0], [output_width-1, 0], [output_width-1, output_height-1], [0, output_height-1]])
            corner_error = float(np.linalg.norm(projected-target, axis=1).max())
            if not math.isfinite(corner_error) or corner_error > 0.02:
                raise ValueError(f"Warp corner mapping error exceeds 0.02 pixels: {corner_error}")
            maximum_corner_error = max(maximum_corner_error, corner_error)
            total += 1
            geometry_types[instance.get("geometry_type", "unknown")] += 1
    if total == 0:
        raise ValueError("No accepted boundary or approximate quad with a verified homography")
    return dict(images=expected, instances=total, empty_images=empty, rejected_instances=rejected,
                geometry_types=dict(geometry_types), boundary_quad_count=geometry_types["boundary_quad"],
                approximate_quad_count=geometry_types["approximate_quad"],
                max_corner_reprojection_error_px=maximum_corner_error,
                boundary_contract="approx_quad_v3", run_json=str((directory / "run.json").resolve()))


def portable_manifest(output):
    """@luna-comment
    Input: Completed package directory with boundary and independent package prediction runs.
    Output: Relocatable manifest containing only package-relative artifact paths.
    Key Steps: Resolve each artifact under package root and rebuild curated relationships.
    Key Params: Reject references escaping package root.
    State/Dependencies: Generated inference records and package files.
    """
    output = output.resolve()
    result = dict(model="best.pt", script="doc_quad_infer.py",
                  requirements="requirements.inference.txt", samples=[], prediction_runs={})
    for path in sorted((output / "samples").iterdir()):
        result["samples"].append(path.resolve().relative_to(output).as_posix())
    for mode in ("predictions", "package_smoke"):
        original = json.loads((output / mode / "run.json").read_text(encoding="utf-8"))
        items = []
        for item in original["results"]:
            path = Path(item["result_json"]).resolve()
            payload = json.loads(path.read_text(encoding="utf-8"))
            items.append(dict(
                source=Path(item["source"]).resolve().relative_to(output).as_posix(),
                result_json=path.relative_to(output).as_posix(),
                overlay=(path.parent / payload["overlay_file"]).resolve().relative_to(output).as_posix(),
                rejected=payload.get("rejected", []),
                instances=[dict(quad=instance["quad"], geometry_type=instance["geometry_type"],
                                prediction_index=instance.get("prediction_index"), instance_id=instance.get("instance_id"),
                                geometry_evidence=instance.get("geometry_evidence", {}), selection_reason=instance.get("selection_reason"),
                                raw_contour_xy=instance["raw_contour_xy"], warp=instance["warp"],
                                crop=(path.parent / instance["image_file"]).resolve().relative_to(output).as_posix())
                           for instance in payload["instances"]]))
        result["prediction_runs"][mode] = dict(run_json=f"{mode}/run.json", results=items)
    result["provenance_note"] = "Original inference JSON retains execution-time provenance; use this manifest to relocate artifacts."
    return result


def render_result_html(output, portable, record):
    """@luna-comment
    Input: Verified package directory, relative artifact manifest and delivery evidence.
    Output: Chinese RESULT.html using only package-relative image and artifact links.
    Key Steps: Escape text, render boundary/contour/warp evidence and explicit rejection reasons.
    Key Params: Rendering requires verified=True; no image base64 embedding.
    State/Dependencies: Validated package artifacts and optional validation.json.
    """
    if record.get("verified") is not True or record.get("boundary_contract") != "approx_quad_v3":
        raise ValueError("Result report requires successful package verification")
    escape = html.escape
    validation_path = output / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None
    sections = []
    modes = {"predictions": ("文档四边形估计与 warp", "inference"),
             "package_smoke": ("交付包独立运行复验", "package_inference")}
    for mode, (label, summary_key) in modes.items():
        summary = record.get(summary_key, {})
        cards = []
        indexed_items = list(enumerate(portable["prediction_runs"][mode]["results"]))
        priorities = {"010": 0, "001": 1}
        indexed_items.sort(key=lambda pair: (priorities.get(Path(pair[1]["source"]).name[:3], 2), pair[0]))
        for index, item in indexed_items:
            payload = json.loads((output / item["result_json"]).read_text(encoding="utf-8"))
            figures = []
            for caption, path in (("原图", item["source"]), ("预测轮廓与拟合四边形", item["overlay"])):
                url = escape(quote(path, safe="/"), quote=True)
                figures.append(f'<figure><a href="{url}"><img src="{url}" loading="lazy" alt="{caption}"></a><figcaption>{caption}</figcaption></figure>')
            coordinates = []
            for instance_index, instance in enumerate(item["instances"]):
                url = escape(quote(instance["crop"], safe="/"), quote=True)
                geometry = ("近似四边形 approximate_quad" if instance["geometry_type"] == "approximate_quad"
                            else "边界拟合 boundary_quad")
                source_index = instance.get("prediction_index")
                figures.append(f'<figure><a href="{url}"><img src="{url}" loading="lazy" alt="区域 {instance_index} 真实单应透视展开"></a><figcaption>区域 {instance_index} · 预测索引 {source_index} · {geometry} · warp</figcaption></figure>')
                coordinates.append(dict(instance=instance_index, prediction_index=source_index, geometry_type=instance["geometry_type"], quad=instance["quad"], raw_contour_xy=instance["raw_contour_xy"], warp=instance["warp"], selection_reason=instance.get("selection_reason"), geometry_evidence=instance.get("geometry_evidence", {})))
            rejection_text = escape(json.dumps(payload.get("rejected", []), ensure_ascii=False, indent=2))
            result_url = escape(quote(item["result_json"], safe="/"), quote=True)
            cards.append(f'<article><h3>样例 {index + 1} · {escape(item["source"])}</h3>'
                         f'<p>检出区域：{len(item["instances"])} · 拒绝区域：{len(payload.get("rejected", []))}</p>'
                         f'<div class="gallery">{"".join(figures)}</div><details><summary>原始轮廓、四角与透视矩阵</summary>'
                         f'<pre>{escape(json.dumps(coordinates, ensure_ascii=False, indent=2))}</pre>'
                         f'<a href="{result_url}">查看原始 JSON</a></details>'
                         f'<details><summary>拒绝原因与原始轮廓（{len(payload.get("rejected", []))}）</summary><pre>{rejection_text}</pre></details></article>')
        sections.append(f'<section id="{mode}"><h2>{label}</h2><p>样例：{summary.get("images", 0)} · '
                        f'区域：{summary.get("instances", 0)} · 空检出图：{summary.get("empty_images", 0)} · '
                        f'边界拟合：{summary.get("boundary_quad_count", 0)} · 近似估计：{summary.get("approximate_quad_count", 0)} · '
                        f'无法处理区域：{summary.get("rejected_instances", 0)}</p>'
                        f'<pre>{escape(json.dumps(summary.get("geometry_types", {}), ensure_ascii=False, indent=2))}</pre>'
                        f'{"".join(cards)}</section>')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>文档检测模型交付结果</title>
<style>body{{font:16px system-ui;background:#f1f5f9;color:#172554;max-width:1500px;margin:30px auto;padding:0 20px}}
section,header,article{{background:white;padding:20px;border-radius:12px;margin:16px 0}}article{{border:1px solid #dbeafe}}
.gallery{{display:flex;gap:14px;overflow-x:auto;align-items:start}}figure{{margin:0;flex:0 0 30%;min-width:220px}}
img{{width:100%;height:330px;object-fit:contain;background:#e2e8f0}}figcaption{{padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
a{{color:#2563eb}}nav a{{margin-right:18px}}@media(max-width:700px){{figure{{flex-basis:85%}}}}</style></head><body>
<header><h1>文档四边形估计与 warp 交付结果</h1><p>已验收边界拟合或近似估计四边形、真实单应变换四角映射、输出尺寸和交付包独立推理。近似结果单独标明；此结论不等于应用效果达标。</p>
<p>检查点：<a href="best.pt">best.pt</a> · SHA256：<code>{escape(record["checkpoint_sha256"])}</code></p>
<p><a href="portable_manifest.json">相对路径产物清单</a> · <a href="README.md">运行说明</a> · <a href="requirements.inference.txt">推理依赖版本</a></p>
<nav><a href="#predictions">边界与透视展开</a><a href="#package_smoke">包内复验</a></nav>
<p>点击图片查看原尺寸；区域较多时可横向滚动。样例 010、001 优先展示，全部样例与拒绝记录均保留。整包移动后应保留目录结构。</p></header>
<section><h2>验证指标</h2><pre>{escape(json.dumps({"validation": validation, "test": record.get("test_metrics")}, ensure_ascii=False, indent=2))}</pre>
<p>缺失指标显示 null。抽样推理通过不代表所有文档均能检出，空检出和拒绝数量如实保留。</p></section>
<section><h2>结果解释与限制</h2><p>优先拟合分割轮廓的四条边；非四边形或边界支持不足时，用轮廓凸包的旋转包络估计 approximate_quad，并同样执行单应 warp。最终 poly 始终是四条直线；轮廓曲线仅用于诊断。</p>
<p>图例：绿色四边形为边界拟合，青色四边形为近似估计；紫色原始轮廓仅为诊断信息，不是最终检测 poly。近似估计是有效输出，按类别单独统计。</p>
<p>近似四边形主要校正整体方向与范围，不保证真实物理四角、文字完整覆盖或宽高比例。曲线、折痕不能完全展平。画外角点保留，不硬截断；warp 对未知画外部分填白，不恢复隐藏内容。--crop-only 仍只是另行诊断裁剪。</p>
<p>几何校正通过不代表实例完整或文字语义正向。模型可能把同一页的可见碎片识别成多个实例；多个碎片分别产生四边形，不等于恢复了整页。原始预测索引和所有拒绝原因均保留。</p>
<p>书页弯曲、折扇折痕、遮挡和画外内容无法由单一透视变换恢复。没有进行文字增强或遮挡文字补全。区域坐标以原图像素为单位。</p></section>
{"".join(sections)}</body></html>'''
    (output / "RESULT.html").write_text(page, encoding="utf-8")


def main():
    """@luna-comment
    Input: Training run, prepared dataset, inference script directory and output CLI.
    Output: Copied model/examples/code, package manifest and truthful verification record.
    Key Steps: Infer held-out samples, validate pixels/geometry, evaluate, package and independently rerun package.
    Key Params: Default twelve samples; optional test evaluation can be disabled explicitly.
    State/Dependencies: Trained local checkpoint, Ultralytics, subprocess Python and filesystem.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run-dir", "data-dir", "source-dir", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--no-eval-test", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    run, data, output = args.run_dir.resolve(), args.data_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = dict(verified=False, stage="verifying", started_at=time.time(), output=str(output),
                  boundary_contract="approx_quad_v3")
    write_json(run / "delivery_verification.json", record)
    try:
        from ultralytics import YOLO
        best = run / "weights/best.pt"
        if not best.is_file() or best.stat().st_size == 0:
            raise ValueError("A nonempty local best.pt is required")
        model = YOLO(str(best))
        if model.task != "segment":
            raise ValueError("Checkpoint is not a segmentation model")
        records = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        selected = select_samples(records, args.samples)
        samples = output / "samples"
        if samples.exists() and any(samples.iterdir()):
            raise ValueError("Output samples directory is nonempty; use a fresh delivery output")
        samples.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(selected):
            source = Path(row["prepared_image"])
            destination = samples / f"{index:03d}_{source.name}"
            shutil.copy2(source, destination)
            row["delivery_sample"] = str(destination)
        write_json(output / "sample_manifest.json", selected)
        script = args.source_dir.resolve() / "doc_quad_infer.py"
        command = [sys.executable, str(script), "--weights", str(best), "--source", str(samples),
                   "--output", str(output / "predictions"), "--device", args.device]
        with (output / "inference.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        geometry = verify_predictions(output / "predictions", len(selected))
        record.update(stage="evaluating_test", inference=geometry)
        write_json(run / "delivery_verification.json", record)
        test_metrics = None
        if not args.no_eval_test and any(row["split"] == "test" for row in records):
            raw = model.val(data=str(data / "data.yaml"), split="test", device=args.device,
                            imgsz=640, batch=16, workers=4, project=str(output),
                            name="test_validation", exist_ok=True).results_dict
            test_metrics = {str(key): float(value) for key, value in raw.items()}
            if not test_metrics or not all(math.isfinite(v) for v in test_metrics.values()):
                raise ValueError("Test validation metrics missing or nonfinite")
            write_json(output / "test_validation.json", test_metrics)
        destination = output / "best.pt"
        shutil.copy2(best, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        shutil.copy2(script, output / script.name)
        package_command = [sys.executable, "-I", "doc_quad_infer.py", "--weights", "best.pt",
                           "--source", "samples", "--output", "package_smoke", "--device", args.device]
        with (output / "package_smoke.log").open("w", encoding="utf-8") as log:
            subprocess.run(package_command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        package_geometry = verify_predictions(output / "package_smoke", len(selected))
        requirements = []
        for distribution in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python"):
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                if distribution != "opencv-python":
                    raise
                distribution = "opencv-python-headless"
                version = importlib.metadata.version(distribution)
            requirements.append(f"{distribution}=={version}")
        (output / "requirements.inference.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                                capture_output=True, text=True).stdout
        (output / "requirements.freeze.txt").write_text(freeze, encoding="utf-8")
        if (run / "validation.json").is_file():
            shutil.copy2(run / "validation.json", output / "validation.json")
        readme = """# Document segmentation inference delivery

Run from this package directory with the recorded Python environment. The five direct
inference dependencies and their installed versions are in requirements.inference.txt;
requirements.freeze.txt records the full environment. Install a torch/torchvision
build appropriate to the destination CUDA/CPU platform.

    python doc_quad_infer.py --weights best.pt --source samples --output new_predictions --device 0

Use --device cpu for CPU inference. Default results are boundary_quad instances fitted
to predicted segmentation contours, or approximate_quad rotated hull envelopes when
four-edge boundary support is insufficient. Both define a real homography;
the image is warped to its recorded output dimensions. Raw contours, fitted quads,
geometry evidence and warp matrices are retained in per-image result.json. Overlays
show raw contours and accepted quads separately. Approximate results retain the failed
boundary-fit reason and are counted separately. Non-quadrilateral documents may therefore
still receive an estimated poly and warp; invalid or degenerate contours remain rejected.
No text enhancement or hidden-text recovery is performed.
The copied script and copied checkpoint were independently executed from this package
directory with Python isolated mode (-I), without repository/PYTHONPATH imports.
Their verified outputs are under package_smoke. portable_manifest.json contains
package-relative artifact relationships for relocation. The original per-image/run
JSON records retain execution-time absolute paths as provenance.

The optional --crop-only mode produces diagnostic_crop only. It is excluded from this
delivery acceptance. --strict-boundary disables approximate fallback for diagnosis.
Approximate warps correct overall orientation and range; they cannot fully flatten curved
or folded pages or guarantee physical corners, complete coverage or metric aspect ratio.
Out-of-frame corners are preserved and unknown pixels padded white, explicitly recorded
in warp metadata. Hidden content is not reconstructed. Inspect RESULT.html for both types.

The included samples are selected from held-out test data (validation only if test
is unavailable), stratified by source. Passing artifact/geometry verification is
not a guarantee of document recall or application-level quality. Review the recorded
empty/rejected counts, validation metrics and example images before deployment.
"""
        (output / "README.md").write_text(readme, encoding="utf-8")
        record.update(verified=True, stage="completed", finished_at=time.time(),
                      checkpoint=str(destination), checkpoint_sha256=digest,
                      inference=geometry,
                      package_inference=package_geometry, package_command=package_command,
                      test_metrics=test_metrics, sample_split=selected[0]["split"],
                      python=sys.version, inference_command=command)
        portable = portable_manifest(output)
        write_json(output / "portable_manifest.json", portable)
        render_result_html(output, portable, record)
        write_json(output / "delivery_manifest.json", record)
        write_json(run / "delivery_verification.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    except BaseException as error:
        record.update(verified=False, stage="failed", finished_at=time.time(),
                      error=str(error), traceback=traceback.format_exc())
        write_json(output / "delivery_manifest.json", record)
        write_json(run / "delivery_verification.json", record)
        raise


if __name__ == "__main__":
    main()
