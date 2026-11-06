"""Persistent, read-only task observation sources. Never executes source HTML or jobs."""
import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("Y:/guang.yang/TextSR/data")
MAX_JSON = 32 * 1024 * 1024


def timestamp():
    """@luna-comment Input: none. Output: UTC ISO timestamp. Key Steps: format current clock.
    Key Params: UTC only. State/Dependencies: datetime; no job-state inference.
    """
    return datetime.now(timezone.utc).isoformat()


def local_path(value, base=None):
    """@luna-comment Input: local absolute path, file URI or manifest-relative raster reference.
    Output: resolved local Path. Key Steps: decode file URI, reject remote URL/UNC, resolve base.
    Key Params: file URI host must be empty/localhost. State/Dependencies: pathlib only; no downloads.
    """
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        raise ValueError("Invalid local path")
    if value.startswith("file:"):
        uri = urlsplit(value)
        if uri.netloc not in ("", "localhost"):
            raise ValueError("Remote file hosts are unsupported")
        value = unquote(uri.path)
        if re.match(r"^/[A-Za-z]:/", value):
            value = value[1:]
    elif "://" in value:
        raise ValueError("Only local files are supported")
    if value.startswith(("\\\\", "//")):
        raise ValueError("Use a registered local/mapped-drive path")
    path = Path(value)
    if not path.is_absolute():
        if base is None:
            raise ValueError("Use an absolute source path")
        path = base / path
    return path.resolve()


class MonitorMarkup(HTMLParser):
    """Inert legacy card extractor; source scripts/styles are never interpreted."""
    def __init__(self):
        """@luna-comment Input: none. Output: empty article/data/GPU extractor.
        Key Steps: initialize parser fields. Key Params: ignore executable text.
        State/Dependencies: standard HTMLParser with character-reference decoding.
        """
        super().__init__(convert_charrefs=True)
        self.cards, self.current, self.field = [], None, None
        self.heading, self.section, self.skip, self.gpu = "", "training", 0, ""
        self.pre_gpu = False

    def handle_starttag(self, tag, attrs):
        """@luna-comment Input: parsed start tag. Output: inert card/progress collection state.
        Key Steps: ignore script/style; select article text and numeric progress.
        Key Params: no links/scripts copied. State/Dependencies: parser fields only.
        """
        attributes = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        if self.skip:
            return
        if tag == "h2":
            self.heading, self.field = "", "h2"
        elif tag == "article":
            self.current = dict(label="", status="", details=[], kind=self.section)
        elif self.current is not None and tag in ("h3", "b", "p", "pre"):
            self.field = tag
            if tag in ("p", "pre"):
                self.current["details"].append("")
        elif self.current is not None and tag == "progress":
            try:
                self.current["current"] = float(attributes.get("value", "0"))
                self.current["total"] = float(attributes.get("max", "100"))
            except ValueError:
                pass
        elif tag == "pre" and self.section == "gpu":
            self.pre_gpu = True

    def handle_endtag(self, tag):
        """@luna-comment Input: parsed closing tag. Output: finalized card/section state.
        Key Steps: close ignored text; classify section headings; append completed articles.
        Key Params: source status is literal. State/Dependencies: parser fields only.
        """
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        if self.skip:
            return
        if tag == "h2":
            if "数据" in self.heading:
                self.section = "data"
            elif "GPU" in self.heading:
                self.section = "gpu"
            self.field = None
        elif tag == "article" and self.current is not None:
            self.cards.append(self.current)
            self.current, self.field = None, None
        elif tag == self.field:
            self.field = None
        if tag == "pre":
            self.pre_gpu = False

    def handle_data(self, data):
        """@luna-comment Input: decoded text. Output: plain labels/details, heading or GPU text.
        Key Steps: route only selected text; never evaluate source expressions.
        Key Params: source strings remain data. State/Dependencies: parser fields.
        """
        if self.skip:
            return
        if self.field == "h2":
            self.heading += data
        elif self.current is not None:
            if self.field == "h3":
                self.current["label"] += data
            elif self.field == "b":
                self.current["status"] += data
            elif self.field in ("p", "pre") and self.current["details"]:
                self.current["details"][-1] += data
        elif self.pre_gpu:
            self.gpu += data


def monitor_data(path):
    """@luna-comment Input: registered JSON or legacy HTML source. Output: normalized snapshot.
    Key Steps: parse JSON or inert markup/strict embedded JSON; bound real-step loss histories.
    Key Params: max2000 observed points per series, retain endpoint; downsampling labeled.
    State/Dependencies: source file only, never source script execution or remote job control.
    """
    if path.stat().st_size > MAX_JSON:
        raise ValueError("Monitor source exceeds32MiB")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("runs", []), list):
            raise ValueError("Monitor JSON needs a runs array")
        runs, datasets = data.get("runs", []), data.get("datasets", data.get("data", []))
        if not isinstance(datasets, list):
            datasets = [dict(label="Data preparation", details=[json.dumps(datasets, ensure_ascii=False)])]
        result = dict(runs=runs, datasets=datasets, updated=data.get("updated"), gpu=str(data.get("gpu", "")))
        history = data.get("lossHistory", {})
    else:
        parser = MonitorMarkup()
        parser.feed(text)
        match = re.search(r"(?:const|let|var)\s+lossHistory\s*=\s*", text)
        history = json.JSONDecoder().raw_decode(text[match.end():])[0] if match else {}
        updated = re.search(r"更新[：:]\s*([^<]+)", text)
        result = dict(runs=[x for x in parser.cards if x["kind"] == "training"],
                      datasets=[x for x in parser.cards if x["kind"] == "data"],
                      updated=updated[1].strip() if updated else None, gpu=parser.gpu)
    if not isinstance(history, dict):
        raise ValueError("lossHistory must be an object")
    histories, counts = {}, {}
    for label, records in history.items():
        if not isinstance(records, list):
            continue
        valid = [row for row in records if isinstance(row, dict) and isinstance(row.get("step"), (int, float)) and
                 math.isfinite(row["step"]) and isinstance(row.get("values"), dict)]
        stride = max(1, math.ceil(len(valid) / 2000))
        sampled = valid[::stride]
        if valid and (not sampled or sampled[-1] is not valid[-1]):
            sampled.append(valid[-1])
        histories[label] = [dict(step=row["step"], values={key: value for key, value in row["values"].items()
                              if isinstance(value, (int, float)) and math.isfinite(value)}) for row in sampled]
        counts[label] = dict(source=len(valid), displayed=len(sampled))
    result.update(lossHistory=histories, historyCounts=counts)
    return result


def gallery_data(path):
    """@luna-comment Input: registered manifest JSON. Output: validated gallery data and asset map.
    Key Steps: preserve stable row/model keys and notes; require unique IDs and supported schema.
    Key Params: max10000 rows/64 models; no image decoding or inference.
    State/Dependencies: manifest file only; actual raster paths validated at serving time.
    """
    if path.stat().st_size > MAX_JSON:
        raise ValueError("Gallery manifest exceeds32MiB")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    models, rows = data.get("models"), data.get("rows")
    if not isinstance(models, list) or not isinstance(rows, list) or not 1 <= len(models) <= 64 or not 1 <= len(rows) <= 10000:
        raise ValueError("Manifest needs1..64 models and1..10000 rows")
    if any(not isinstance(item, dict) for item in models + rows):
        raise ValueError("Models and rows must be objects")
    keys = [item.get("key") for item in models]
    ids = [str(item.get("id", "")) for item in rows]
    if any(not isinstance(key, str) or not key or len(key) > 150 for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Model keys must be unique strings")
    if any(not value or len(value) > 150 for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Sample IDs must be unique")
    assets = {}
    normalized = []
    for row, sample in zip(rows, ids):
        if any(not isinstance(row.get(key), int) or isinstance(row.get(key), bool) or not 1 <= row[key] <= 100000 for key in ("width", "height")):
            raise ValueError("Each sample needs positive native width and height")
        images = dict(row.get("images", {}))
        if row.get("lq"):
            images["lq"] = row["lq"]
        assets[sample] = {key: value for key, value in images.items() if key in keys and isinstance(value, str)}
        normalized.append(dict(id=sample, name=str(row.get("name", sample)), width=row.get("width"),
                               height=row.get("height"), available=list(assets[sample])))
    return dict(title=str(data.get("title", "Gallery")), notes=str(data.get("notes", "")),
                defaults=data.get("defaults", ""), models=models, rows=normalized, _assets=assets)


def read_source(path, kind):
    """@luna-comment Input: approved source path/kind. Output: stable data/stamp/stat tuple.
    Key Steps: bind metadata before/after parsing; retry once on concurrent publication.
    Key Params: two attempts, never cache mismatched generations.
    State/Dependencies: source read only; active publisher is never interrupted.
    """
    for _ in range(2):
        before = path.stat()
        data = (gallery_data if kind == "gallery" else monitor_data)(path)
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return data, str((after.st_mtime_ns, after.st_size)), after
    raise ValueError("Source is being updated; retry on next refresh")


class ResourceStore:
    def __init__(self, database, root=ROOT, allowed_roots=None):
        """@luna-comment Input: local registry DB/root and optional fixture boundaries.
        Output: durable source/snapshot/preset registry. Key Steps: create independent schema.
        Key Params: source roots never include user/system directories by default.
        State/Dependencies: RLock/SQLite; never touches original task execution state.
        """
        self.root = Path(root).resolve()
        self.allowed = [Path(value).resolve() for value in (allowed_roots or [self.root, ARTIFACT_ROOT])]
        self.lock = threading.RLock()
        database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS resources(id TEXT PRIMARY KEY,data TEXT NOT NULL,stamp TEXT,snapshot TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS views(resource TEXT,name TEXT,state TEXT,PRIMARY KEY(resource,name))")
        self.db.commit()

    def allowed_path(self, value, base=None):
        """@luna-comment Input: explicit path or manifest reference/base. Output: approved local Path.
        Key Steps: canonicalize and contain within configured read roots.
        Key Params: no drive-root or arbitrary system/private file access.
        State/Dependencies: self.allowed; filesystem reads only.
        """
        roots = [root.resolve() for root in self.allowed]
        # Reject unknown UNC hosts lexically before any resolution/contact.
        if isinstance(value, str) and value.startswith(("\\\\", "//")):
            candidate = Path(value)
            if not any(candidate.is_relative_to(root) for root in roots):
                raise ValueError("Unapproved network path")
            path = candidate.resolve()
        else:
            path = local_path(value, base)
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError("Source is outside the approved workspace/artifact roots")
        return path

    def tasks(self, issued=()):
        """@luna-comment Input: existing issued records. Output: selectable real Markdown task refs.
        Key Steps: enumerate repository task files and local issuance docs, read titles only.
        Key Params: no ownership or runtime-state write. State/Dependencies: repository/task and records.
        """
        result = []
        paths = sorted((self.root / "task").glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths:
            if "template" in path.name.lower():
                continue
            if not path.resolve().is_relative_to(self.root):
                continue
            with path.open(encoding="utf-8-sig") as stream:
                heading = stream.readline().strip().lstrip("# ")
            result.append(dict(ref=path.relative_to(self.root).as_posix(), title=heading or path.stem))
        for task in issued:
            value = task.get("recordFile")
            if value:
                path = Path(value).resolve()
                if path.is_relative_to(self.root) and path.is_file():
                    result.append(dict(ref=path.relative_to(self.root).as_posix(), title=task.get("metadata", {}).get("title", task["id"])))
        return result

    def list(self):
        """@luna-comment Input: none. Output: registry metadata without snapshots/assets.
        Key Steps: read durable ordered entries. Key Params: no source I/O.
        State/Dependencies: locked SQLite only.
        """
        with self.lock:
            return [json.loads(row[0]) for row in self.db.execute("SELECT data FROM resources ORDER BY rowid DESC")]

    def get(self, resource_id):
        """@luna-comment Input: registry ID. Output: metadata or KeyError.
        Key Steps: parameterized lookup. Key Params: no path-based lookup.
        State/Dependencies: locked SQLite.
        """
        with self.lock:
            row = self.db.execute("SELECT data FROM resources WHERE id=?", (resource_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown observation source")
            return json.loads(row[0])

    def register(self, payload, tasks):
        """@luna-comment Input: reviewed kind/title/taskRef/source/projects/defaults and valid tasks.
        Output: durable idempotent registration. Key Steps: validate scope/schema, insert atomically.
        Key Params: writes only own registry; source and original task files unchanged.
        State/Dependencies: self.allowed_path/gallery_data/monitor_data and locked SQLite.
        """
        kind, title, task_ref = payload.get("kind"), payload.get("title"), payload.get("taskRef")
        if kind not in ("monitor", "gallery") or not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
            raise ValueError("Choose type and enter a title (max200 characters)")
        if task_ref not in {task["ref"] for task in tasks}:
            raise ValueError("Select an existing task")
        path = self.allowed_path(payload.get("source"))
        if kind == "gallery" and path.suffix.lower() == ".html":
            path = self.allowed_path(str(path.with_name("manifest.json")))
        if path.suffix.lower() not in ((".json",) if kind == "gallery" else (".json", ".html")) or not path.is_file():
            raise ValueError("Choose an existing supported JSON manifest or monitor source")
        projects, defaults = payload.get("projects", []), payload.get("defaults", {})
        if not isinstance(projects, list) or len(projects) > 20 or any(not isinstance(item, str) or len(item) > 200 for item in projects):
            raise ValueError("Invalid project labels")
        if not isinstance(defaults, dict) or len(json.dumps(defaults)) > 20000:
            raise ValueError("Invalid default view")
        snapshot, stamp, source_stat = read_source(path, kind)
        if not snapshot.get("runs") and kind == "monitor" and not snapshot.get("datasets"):
            raise ValueError("Monitor contains no run/data records")
        with self.lock:
            for existing in self.list():
                if existing["kind"] == kind and existing["taskRef"] == task_ref and existing["source"].casefold() == str(path).casefold():
                    return existing
            resource = dict(id=uuid.uuid4().hex, kind=kind, title=title.strip(), taskRef=task_ref,
                            source=str(path), projects=projects, defaults=defaults, createdAt=timestamp())
            snapshot.update(observedAt=timestamp(), sourceModifiedAt=datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat())
            self.db.execute("INSERT INTO resources VALUES(?,?,?,?)", (resource["id"], json.dumps(resource, ensure_ascii=False),
                            stamp, json.dumps(snapshot, ensure_ascii=False, allow_nan=False)))
            self.db.commit()
            return resource

    def snapshot(self, resource_id, private=False):
        """@luna-comment Input: registered ID and internal asset-map flag. Output: current/stale snapshot.
        Key Steps: stat source, parse changed data, atomically cache, label any read failure.
        Key Params: stale cached data never claims live execution. State/Dependencies: locked registry/source.
        """
        resource = self.get(resource_id)
        with self.lock:
            stamp, saved = self.db.execute("SELECT stamp,snapshot FROM resources WHERE id=?", (resource_id,)).fetchone()
            cached = json.loads(saved) if saved else {}
            error = None
            try:
                path = self.allowed_path(resource["source"])
                stat = path.stat()
                current = str((stat.st_mtime_ns, stat.st_size))
                if current != stamp:
                    cached, current, source_stat = read_source(path, resource["kind"])
                    cached.update(observedAt=timestamp(), sourceModifiedAt=datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat())
                    self.db.execute("UPDATE resources SET stamp=?,snapshot=? WHERE id=?", (current, json.dumps(cached, ensure_ascii=False, allow_nan=False), resource_id))
                    self.db.commit()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            result = dict(cached)
            if not private:
                result.pop("_assets", None)
            result.update(resource=resource, stale=error is not None, error=error, checkedAt=timestamp(), views=self.views(resource_id))
            return result

    def asset(self, resource_id, sample, model):
        """@luna-comment Input: registered gallery/sample/model IDs. Output: approved exact raster Path.
        Key Steps: resolve only stored manifest mapping, then enforce canonical root/type/size.
        Key Params: no arbitrary file route, HTML/SVG/URL or interpolation parameters.
        State/Dependencies: snapshot and read-only filesystem; original bytes served unchanged.
        """
        resource = self.get(resource_id)
        if resource["kind"] != "gallery":
            raise ValueError("Not a gallery")
        data = self.snapshot(resource_id, private=True)
        value = data["_assets"][sample][model]
        path = self.allowed_path(value, Path(resource["source"]).parent)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp") or not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("Missing or unsupported raster asset")
        return path

    def views(self, resource_id):
        """@luna-comment Input: resource ID. Output: named durable view states.
        Key Steps: parameterized read. Key Params: states never execute actions.
        State/Dependencies: locked SQLite.
        """
        with self.lock:
            return [dict(name=row[0], state=json.loads(row[1])) for row in self.db.execute("SELECT name,state FROM views WHERE resource=? ORDER BY name", (resource_id,))]

    def save_view(self, resource_id, payload):
        """@luna-comment Input: reviewed name and JSON view state. Output: saved preset list.
        Key Steps: validate resource/state limits, upsert and commit.
        Key Params: bounded JSON, not source paths/scripts. State/Dependencies: locked registry only.
        """
        self.get(resource_id)
        name, state = payload.get("name"), payload.get("state")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100 or not isinstance(state, dict):
            raise ValueError("Enter a preset name and valid view state")
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        if len(encoded) > 20000:
            raise ValueError("View state too large")
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO views VALUES(?,?,?)", (resource_id, name.strip(), encoded))
            self.db.commit()
        return self.views(resource_id)
