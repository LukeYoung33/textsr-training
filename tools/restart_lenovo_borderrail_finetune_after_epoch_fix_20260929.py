"""Restart only active finetune trainers so they load the corrected loop."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path


STATUS = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_continuous_20260928/status.json")


def main() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    targets = []
    for name, row in status["jobs"].items():
        if not name.startswith("finetune_") or row.get("state") != "running":
            continue
        pid = int(row["pid"])
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if "realesrgan/train.py" not in command or name not in command:
            raise RuntimeError(f"refusing to signal PID {pid}: {command}")
        targets.append((name, pid))
    if len(targets) != 8:
        raise RuntimeError(f"expected eight active finetunes, got {len(targets)}")
    for name, pid in targets:
        os.kill(pid, signal.SIGTERM)
        print(f"signaled {name}: {pid}", flush=True)


if __name__ == "__main__":
    main()
