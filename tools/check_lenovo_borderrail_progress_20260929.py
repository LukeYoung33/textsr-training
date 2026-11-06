"""Summarize current Lenovo border-rail training progress from logs and files."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_borderrail_continuous_20260928")
PATTERN = re.compile(r"iter:\s*([\d,]+).*?eta:\s*([^,]+),")


def main() -> None:
    status = json.loads((ROOT / "status.json").read_text(encoding="utf-8"))
    rows = []
    for name, item in status["jobs"].items():
        if not name.startswith("finetune_"):
            continue
        log = Path(item.get("log", ""))
        iteration = 0
        eta = None
        if log.is_file():
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                match = PATTERN.search(line)
                if match:
                    iteration = int(match.group(1).replace(",", ""))
                    eta = match.group(2)
        experiment = Path(item["experiment"])
        complete = all((experiment / filename).is_file() for filename in (
            "models/net_g_100000.pth", "models/net_g_latest.pth", "training_states/100000.state"))
        rows.append({"name": name, "state": item["state"], "iteration": iteration,
                     "eta_log": eta, "complete_checkpoint": complete,
                     "attempt": item.get("attempt"), "last_error": item.get("last_error")})
    print(json.dumps({"stage": status["stage"], "updated_at": status["updated_at"], "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
