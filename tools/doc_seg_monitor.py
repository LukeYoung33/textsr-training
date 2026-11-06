"""Render an independent, self-contained Chinese training status page."""
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import subprocess
import time


def read_json(path):
    """@luna-comment
    Input: Optional JSON path.
    Output: Mapping or empty mapping on unavailable/malformed file.
    Key Steps: Decode complete snapshot.
    Key Params: None.
    State/Dependencies: Read-only filesystem.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def alive(state):
    """@luna-comment
    Input: Training state including PID and optional Linux start ticks.
    Output: Boolean matching live process status.
    Key Steps: Probe process, reject reused PID and zombies on Linux.
    Key Params: process_start protects against PID reuse.
    State/Dependencies: OS process table.
    """
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            return fields[0] != "Z" and (not state.get("process_start") or fields[19] == state["process_start"])
        return True
    except (OSError, IndexError):
        return False


def chart(rows, keys, title):
    """@luna-comment
    Input: Numeric CSV rows, metric keys and display title.
    Output: Inline SVG with independently scaled shared y axis.
    Key Steps: Filter finite pairs, derive bounds and plot polylines.
    Key Params: Each chart shares its own scale; x is epoch.
    State/Dependencies: No network or plotting package.
    """
    colors = ["#2563eb", "#ef4444", "#16a34a", "#a855f7"]
    series = []
    for key in keys:
        points = []
        for row in rows:
            try:
                point = (float(row["epoch"]), float(row[key]))
                if all(math.isfinite(v) for v in point):
                    points.append(point)
            except (KeyError, ValueError, TypeError):
                continue
        if points:
            series.append((key, points))
    if not series:
        return f"<h3>{html.escape(title)}</h3><p>等待首轮指标</p>"
    max_x = max(p[0] for _, points in series for p in points)
    values = [p[1] for _, points in series for p in points]
    low, high = min(0, min(values)), max(values)
    span = max(high - low, 1e-9)
    pieces = [f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 650 250" role="img">',
              '<path d="M55 15V215H635" fill="none" stroke="#94a3b8"/>',
              f'<text x="1" y="22">{high:.4g}</text><text x="1" y="215">{low:.4g}</text>',
              f'<text x="55" y="240">epoch 0</text><text x="570" y="240">{max_x:g}</text>']
    for index, (key, points) in enumerate(series):
        coords = " ".join(f"{55 + 575*x/max(max_x,1):.2f},{215-195*(y-low)/span:.2f}" for x,y in points)
        pieces.append(f'<polyline points="{coords}" fill="none" stroke="{colors[index%4]}" stroke-width="2"/>')
    pieces.append("</svg>")
    for index, (key, points) in enumerate(series):
        pieces.append(f'<span style="color:{colors[index%4]}">{html.escape(key)}: {points[-1][1]:.5g}</span> ')
    return "".join(pieces)


def render(run):
    """@luna-comment
    Input: Run directory containing status, CSV and optional audit/delivery records.
    Output: Atomic STATUS.html and monitor_status.json.
    Key Steps: Independently inspect process, freshness, metrics, GPU and artifacts.
    Key Params: Heartbeat stale threshold 120 seconds.
    State/Dependencies: nvidia-smi optional; HTML has no external resources.
    """
    now = time.time()
    state = read_json(run / "status.json")
    verification = read_json(run / "delivery_verification.json")
    running = alive(state)
    age = now - float(state.get("heartbeat_at", 0))
    stage = state.get("stage", "waiting")
    terminal = stage in ("failed", "training_complete")
    if not terminal and state and not running:
        display = "训练进程已退出，未确认完成"
    elif not terminal and state and age > 120:
        display = "心跳过期，请检查训练"
    else:
        display = {"setup":"环境/数据初始化", "running":"训练中", "validating":"最终验证中",
                   "training_complete":"训练完成，待推理交付验收", "failed":"训练失败"}.get(stage, "等待启动")
    if stage == "training_complete" and verification.get("verified") is True:
        display = ("训练完成，四边形估计与 warp 交付已验收（含明确标记的近似结果）"
                   if verification.get("boundary_contract") == "approx_quad_v3" else
                   "训练完成；此前交付为历史版本，等待新版四边形估计与 warp 验收")
    elif stage == "training_complete" and verification.get("stage") == "failed":
        display = "训练完成，推理交付验收失败：" + str(verification.get("error", "请查看验收记录"))
    elif stage == "training_complete" and verification.get("stage") in (
            "verifying", "verifying_perspective", "evaluating_test"):
        display = "训练完成，推理交付验收中"
    rows = []
    try:
        with (run / "results.csv").open(encoding="utf-8") as file:
            rows = [{k.strip():v.strip() for k,v in row.items() if k and v is not None}
                    for row in csv.DictReader(file)]
    except OSError:
        pass
    config = state.get("config", {})
    epoch = state.get("epoch", 0)
    budget = config.get("epochs", "?")
    elapsed = max(0, state.get("finished_at", now) - state.get("started_at", now))
    eta = max(0, (budget - epoch) * elapsed / epoch) if isinstance(budget, int) and epoch else None
    gpu = "GPU 信息暂不可用"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                              "--format=csv"], capture_output=True, text=True, timeout=8).stdout or gpu
    except (OSError, subprocess.TimeoutExpired):
        pass
    log = ""
    for name in ("train.log", "stdout.log"):
        try:
            with (run / name).open("rb") as file:
                file.seek(0, 2)
                file.seek(max(0, file.tell() - 18000))
                log = file.read().decode("utf-8", errors="replace")
            break
        except OSError:
            pass
    checkpoints = []
    for name in ("best.pt", "last.pt"):
        path = run / "weights" / name
        checkpoints.append(f"{path}: " + (f"{path.stat().st_size/1048576:.2f} MiB" if path.exists() else "尚未生成"))
    audit = read_json(run / "dataset_audit.json")
    if not audit and config.get("data"):
        audit = read_json(Path(config["data"]).parent / "audit.json")
    escape = html.escape
    charts = chart(rows, ["train/box_loss","train/seg_loss","val/box_loss","val/seg_loss"], "检测与分割损失")
    charts += chart(rows, ["metrics/mAP50(B)","metrics/mAP50-95(B)","metrics/mAP50(M)","metrics/mAP50-95(M)"], "验证集 Box / Mask mAP")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30">
<title>文档 YOLO11n-seg 训练监控</title><style>
body{{font:16px system-ui;background:#f1f5f9;color:#172554;margin:24px auto;padding:0 20px;max-width:1200px}}
section{{background:white;border-radius:12px;padding:20px;margin:18px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px}}
svg{{width:100%;max-height:320px}}span{{display:inline-block;margin:6px}}.badge{{background:#dbeafe;padding:12px;border-radius:8px}}h1{{font-size:28px}}
</style><h1>文档 YOLO11n-seg 训练监控</h1>
<p>页面每 30 秒刷新；生成时间：{time.strftime('%Y-%m-%d %H:%M:%S %Z')}。离线快照不会自动获取远端更新。</p>
<p id="freshness"></p><button onclick="location.reload()">立即刷新</button>
<section><h2 class="badge">{escape(display)}</h2><p>已完成 epoch：{epoch} / {budget}（可早停） · PID：{state.get('pid','—')} · 进程存活：{running}</p>
<p>心跳年龄：{age:.0f} 秒 · 已用时：{elapsed/3600:.2f} 小时 · 预计剩余：{f'{eta/3600:.2f} 小时' if eta is not None and not terminal else '—'}</p>
<p>ETA 按已完成轮次粗估，不代表完成承诺。进程存活不代表训练有进展。</p></section>
<section>{charts}</section><section><h2>GPU 当前状态</h2><pre>{escape(gpu)}</pre></section>
<section><h2>检查点</h2><pre>{escape(chr(10).join(checkpoints))}</pre><p>存在检查点不等于训练已完成；最终交付以独立推理验收记录为准。</p></section>
<section><h2>超参数</h2><pre>{escape(json.dumps(state.get('effective_train_parameters',config),ensure_ascii=False,indent=2))}</pre>
<h2>数据审计</h2><pre>{escape(json.dumps(audit,ensure_ascii=False,indent=2)) if audit else '尚未读取审计摘要'}</pre></section>
<section><h2>最终验证与交付验收</h2><pre>{escape(json.dumps({'validation':state.get('validation_metrics'), 'delivery':verification},ensure_ascii=False,indent=2))}</pre></section>
<section><h2>日志尾部</h2><pre>{escape(log or '暂无日志')}</pre><pre>{escape(state.get('traceback',''))}</pre></section>
<footer>运行目录：{escape(str(run))}。单类区域分割；四边形来自后处理，不能据此认定物理平面。</footer>
<script>function freshness(){{const seconds=Math.max(0,Date.now()/1000-{now});const el=document.getElementById('freshness');
el.textContent=seconds>120?'页面已超过 120 秒未更新：监控可能已停止，以上状态仅为历史快照。':'监控快照年龄：'+Math.floor(seconds)+' 秒';
el.style.color=seconds>120?'#dc2626':'#166534';}}freshness();setInterval(freshness,1000);</script></html>"""
    temporary = run / "STATUS.html.tmp"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(run / "STATUS.html")
    temporary = run / "monitor_status.json.tmp"
    temporary.write_text(json.dumps(dict(generated_at=now, trainer_alive=running, heartbeat_age=age,
                                        display=display, monitor_pid=os.getpid())), encoding="utf-8")
    temporary.replace(run / "monitor_status.json")


def main():
    """@luna-comment
    Input: Run path, optional interval and one-shot flag.
    Output: Repeated status page snapshots until interrupted.
    Key Steps: Render independently then wait; preserve prior HTML on transient errors.
    Key Params: Default 30-second refresh interval, minimum 2 seconds.
    State/Dependencies: Run directory and monitor process.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            render(directory)
        except Exception as error:
            print(f"Monitor render failed: {error}", flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(max(2, args.interval))


if __name__ == "__main__":
    main()
