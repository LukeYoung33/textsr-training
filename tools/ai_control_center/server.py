"""VSCode workspace file/document helper only; no Codex login or inference."""

import argparse
import hashlib
import json
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from documents import Store
from resources import ResourceStore

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "apps" / "ai-control-center"
WORKSPACE = ROOT / "docs" / "Real-ESRGAN.code-workspace"
BACKUP_DIR = ROOT / ".artifacts" / "ai-control-center" / "workspace-backups"
WORKSPACE_LOCK = threading.RLock()
METADATA = {
    "Real-ESRGAN": ("↗", "超分模型开发、训练与验证，图像效果评测及模型导出。", ["模型训练", "效果测试"]),
    "skill-repo-master": ("⇄", "ONNX → TFLite 转换、手机端性能测试与工程技能。", ["模型转换", "Benchmark"]),
    "TextSR_SVN": ("C+", "文字超分 C++ 算法库、图像处理 Pipeline 与工程集成。", ["C++ Pipeline", "算法交付"]),
    "ultralytics": ("⌗", "YOLO 检测与分割训练基础，关联文档检测和矫正任务。", ["文档检测", "检测训练"]),
}


class WorkspaceConflict(ValueError):
    """The editor changed the workspace during a registration attempt."""


def parse_workspace(raw):
    """@luna-comment Input: UTF-8 workspace bytes, optionally BOM/JSONC comments/trailing commas.
    Output: original text, comment-masked text with identical offsets, parsed configuration.
    Key Steps: preserve strings, mask comments, remove only non-string trailing commas, validate folders.
    Key Params: root folders must be local path objects. State/Dependencies: json/re; no filesystem writes.
    """
    text = raw.decode("utf-8-sig")
    masked = list(text)
    position, quoted, escaped = 0, False, False
    while position < len(text):
        char = text[position]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif text.startswith("//", position) or text.startswith("/*", position):
            if text.startswith("//", position):
                end = text.find("\n", position)
                end = len(text) if end == -1 else end
            else:
                end = text.find("*/", position + 2)
                if end == -1:
                    raise ValueError("Workspace contains an unterminated comment.")
                end += 2
            for index in range(position, end):
                if masked[index] not in "\r\n":
                    masked[index] = " "
            position = end
            continue
        position += 1
    clean = "".join(masked)
    strict = re.sub(r'"(?:\\.|[^"\\])*"|,(?=\s*[}\]])',
                    lambda token: token[0] if token[0].startswith('"') else " ", clean)
    config = json.loads(strict)
    if not isinstance(config, dict) or not isinstance(config.get("folders"), list):
        raise ValueError("Workspace must contain a root folders array.")
    if any(not isinstance(folder, dict) or not isinstance(folder.get("path"), str) for folder in config["folders"]):
        raise ValueError("Only local path folders are supported in this workspace.")
    return text, clean, config


def folder_bounds(clean):
    """@luna-comment Input: comment-masked valid workspace text. Output: root folders [ and ] offsets.
    Key Steps: tokenize strings/brackets, identify root key, balance nested containers.
    Key Params: strings never contribute bracket depth. State/Dependencies: re/json only.
    """
    depth, opening = 0, None
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|[{}\[\]]', clean):
        token = match[0]
        if token.startswith('"'):
            if depth == 1 and json.loads(token) == "folders":
                suffix = re.match(r"\s*:\s*\[", clean[match.end():])
                if suffix:
                    opening = match.end() + suffix.end() - 1
        elif token in ("{", "["):
            depth += 1
        else:
            if token == "]" and depth == 2 and opening is not None:
                return opening, match.start()
            depth -= 1
    raise ValueError("Could not locate the root workspace folders array.")


def add_project(payload):
    """@luna-comment Input: explicit absolute existing local directory and optional name.
    Output: refreshed inventory, created flag and exact-byte backup path when created.
    Key Steps: validate, lock/read, deduplicate canonical paths, insert entry into original JSONC,
    validate new text, backup and replace atomically after detecting external edits.
    Key Params: fixed WORKSPACE target; no arbitrary workspace/code writes or project execution.
    State/Dependencies: WORKSPACE_LOCK, BACKUP_DIR, filesystem; never modifies project contents.
    """
    value, name = payload.get("path"), payload.get("name", "")
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Enter an absolute local project directory.")
    if not isinstance(name, str) or len(name) > 200 or any(ord(char) < 32 for char in name):
        raise ValueError("Project display name must be at most 200 characters without control characters.")
    value, name = value.strip(), name.strip()
    path = Path(value)
    if not path.is_absolute() or value.startswith(("\\\\", "//")):
        raise ValueError("Use an absolute local path, not a relative path or network share.")
    if not path.is_dir():
        raise ValueError("The project directory does not exist or is not a directory.")
    path = path.resolve(strict=True)
    if path == Path(path.anchor):
        raise ValueError("Select a project directory rather than an entire drive.")
    with WORKSPACE_LOCK:
        original = WORKSPACE.read_bytes()
        text, clean, config = parse_workspace(original)
        for folder in config["folders"]:
            existing = (WORKSPACE.parent / folder["path"]).resolve()
            if str(existing).casefold() == str(path).casefold():
                return dict(created=False, **load_projects())
        try:
            stored_path = os.path.relpath(path, WORKSPACE.parent).replace("\\", "/")
        except ValueError:
            stored_path = path.as_posix()
        entry = dict(path=stored_path)
        if name:
            entry["name"] = name
        opening, closing = folder_bounds(clean)
        contents = clean[opening + 1:closing]
        prefix = text[:closing]
        if config["folders"] and not contents.rstrip().endswith(","):
            insertion = opening + 1 + len(contents.rstrip())
            prefix = prefix[:insertion] + "," + prefix[insertion:]
        newline = "\r\n" if "\r\n" in text else "\n"
        line_prefix = text[text.rfind("\n", 0, closing) + 1:closing]
        closing_indent = line_prefix if not line_prefix.strip() else "  "
        indent = closing_indent + ("\t" if "\t" in closing_indent else "  ")
        prefix = prefix.rstrip(" \t")
        if not prefix.endswith("\n"):
            prefix += newline
        changed = prefix + indent + json.dumps(entry, ensure_ascii=False) + newline + closing_indent + text[closing:]
        encoded = (b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b"") + changed.encode("utf-8")
        parse_workspace(encoded)
        temporary = WORKSPACE.with_name(WORKSPACE.name + "." + uuid.uuid4().hex + ".tmp")
        backup = BACKUP_DIR / (uuid.uuid4().hex + ".code-workspace")
        try:
            temporary.write_bytes(encoded)
            if WORKSPACE.read_bytes() != original:
                raise WorkspaceConflict("Workspace changed in VSCode. Refresh and retry; no configuration was overwritten.")
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            with backup.open("xb") as stream:
                stream.write(original)
            if WORKSPACE.read_bytes() != original:
                raise WorkspaceConflict("Workspace changed in VSCode. Refresh and retry; no configuration was overwritten.")
            temporary.replace(WORKSPACE)
        finally:
            temporary.unlink(missing_ok=True)
        return dict(created=True, backup=str(backup), **load_projects())


def load_projects():
    """@luna-comment
    Input: fixed, user-confirmed UTF-8 workspace configuration.
    Output: project metadata and resolved paths; no source or credential reads.
    Key Steps: parse JSONC, resolve folder paths, derive stable path IDs, attach curated labels.
    Key Params: METADATA supplies presentation labels, not runtime metrics.
    State/Dependencies: WORKSPACE, pathlib; path existence is a live filesystem check.
    """
    with WORKSPACE_LOCK:
        _, _, config = parse_workspace(WORKSPACE.read_bytes())
    projects = []
    for folder in config["folders"]:
        path = (WORKSPACE.parent / folder["path"]).resolve()
        name = folder.get("name", path.name)
        icon, description, tags = METADATA.get(path.name, ("▦", "工作区登记项目；能力待整理。", ["待分类"]))
        project_id = hashlib.sha256(str(path).casefold().encode("utf-8")).hexdigest()[:20]
        projects.append(dict(id=project_id, name=name, path=str(path), icon=icon,
                             description=description, tags=tags, exists=path.is_dir()))
    return dict(projects=projects, workspace=str(WORKSPACE), connection="disconnected")


class Handler(BaseHTTPRequestHandler):
    def valid_origin(self, mutation=False):
        """@luna-comment Input: request headers and mutation flag. Output: local-origin Boolean.
        Key Steps: validate exact local Host/Origin and fetch-site. Key Params: bound server port.
        State/Dependencies: HTTP headers only; rejects DNS rebinding and cross-site mutations.
        """
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in hosts and (not mutation or (
            (not origin or origin == "http://" + host) and
            self.headers.get("Sec-Fetch-Site") not in ("cross-site", "same-site")))

    def json_response(self, data, status=200):
        """@luna-comment Input: public response object/status. Output: UTF-8 JSON HTTP response.
        Key Steps: serialize and set noncache headers. Key Params: no CORS grants.
        State/Dependencies: request output stream; never includes account credentials.
        """
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        """@luna-comment
        Input: local HTTP GET path; raster queries select registered sample/model IDs only.
        Output: inventory/task/observation JSON, fixed static assets or scoped raster bytes/error.
        Key Steps: dispatch fixed paths, validate registered raster identity and set response headers.
        Key Params: static routes map only to named management/PDM application assets.
        State/Dependencies: APP, load_projects and document Store; GET never executes tasks.
        """
        if not self.valid_origin():
            self.send_error(403)
            return
        url = urlsplit(self.path)
        route = url.path
        try:
            if route == "/api/observation/tasks":
                self.json_response(dict(tasks=self.server.resources.tasks(self.server.store.list("task"))))
                return
            if route == "/api/observation":
                self.json_response(dict(resources=self.server.resources.list()))
                return
            observation = re.fullmatch(r"/api/observation/([a-f0-9]{32})(/asset)?", route)
            if observation:
                try:
                    if observation[2]:
                        query = parse_qs(url.query)
                        path = self.server.resources.asset(observation[1], query.get("sample", [""])[0], query.get("model", [""])[0])
                        self.send_response(200)
                        self.send_header("Content-Type", "image/" + ({".jpg": "jpeg", ".jpeg": "jpeg"}.get(path.suffix.lower(), path.suffix.lower()[1:])))
                        self.send_header("Content-Length", str(path.stat().st_size))
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        with path.open("rb") as stream:
                            while chunk := stream.read(256 * 1024):
                                self.wfile.write(chunk)
                    else:
                        self.json_response(self.server.resources.snapshot(observation[1]))
                except (ValueError, KeyError, FileNotFoundError):
                    self.send_error(404, "Observation resource/asset unavailable")
                return
            if route == "/api/status":
                self.json_response(dict(mode="workspace_frontend",workspace=str(WORKSPACE),execution="external_vscode_codex"))
                return
            if route == "/api/tasks":
                self.json_response(dict(tasks=self.server.store.list("task")))
                return
            if route == "/api/conversations":
                self.json_response(dict(conversations=self.server.store.list("conversation")))
                return
            document = re.fullmatch(r"/api/tasks/([a-f0-9-]{36})/markdown", route)
            if document:
                task = self.server.store.get(document[1])
                if task.get("kind", "task") != "task":
                    self.send_error(404)
                    return
                data = (self.server.store.record_dir / (document[1] + ".md")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="task-{document[1]}.md"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)
                return
            match = re.fullmatch(r"/api/tasks/([a-f0-9-]{36})", route)
            if match:
                task = self.server.store.document(match[1])
                self.json_response(task)
                return
            if route == "/api/projects":
                data = json.dumps(load_projects(), ensure_ascii=False).encode("utf-8")
                mime = "application/json"
            elif route in ("/", "/index.html", "/app.js", "/pdm.html", "/pdm.js", "/pdm-data.json", "/monitor.html", "/compare.html", "/workbench.js", "/workbench.css"):
                filename = "workbench.html" if route in ("/monitor.html", "/compare.html") else "index.html" if route == "/" else route[1:]
                data = (APP / filename).read_bytes()
                mime = {".js": "text/javascript", ".json": "application/json", ".css": "text/css"}.get(Path(filename).suffix, "text/html")
            else:
                self.send_error(404)
                return
        except (OSError, ValueError, KeyError):
            self.send_error(503, "Workspace configuration unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        """@luna-comment Input: same-origin JSON action, max 64 KiB. Output: result/error JSON.
        Key Steps: validate origin/body, dispatch fixed actions, validate task/request ownership.
        Key Params: fixed project/document writes only; AI execution endpoints are disabled.
        State/Dependencies: local workspace config/Store; no subprocess, auth or model calls.
        """
        if not self.valid_origin(mutation=True):
            self.json_response(dict(error="Cross-origin request rejected"), 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json body, maximum 64 KiB.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            route = urlsplit(self.path).path
            preset = re.fullmatch(r"/api/observation/([a-f0-9]{32})/views", route)
            if route == "/api/observation":
                result = self.server.resources.register(payload, self.server.resources.tasks(self.server.store.list("task")))
                self.json_response(result, 201)
            elif preset:
                self.json_response(dict(views=self.server.resources.save_view(preset[1], payload)))
            elif route == "/api/projects":
                result = add_project(payload)
                self.json_response(result, 201 if result["created"] else 200)
            elif route == "/api/tasks":
                self.json_response(self.server.store.create_task(payload,load_projects()), 201)
            else:
                removed = route in ("/api/connect","/api/conversations") or route.endswith(("/interrupt","/respond"))
                self.json_response(dict(error="Independent Codex execution is disabled. Use the existing VSCode Codex panel." if removed else "Unknown action"),410 if removed else 404)
        except WorkspaceConflict as exc:
            self.json_response(dict(error=str(exc)), 409)
        except (ValueError, KeyError, TypeError) as exc:
            self.json_response(dict(error=str(exc)), 400)
        except Exception as exc:
            self.json_response(dict(error=str(exc)), 503)


def seed_observations(resources):
    """@luna-comment Input: current resource registry. Output: supplied sources registered once if available.
    Key Steps: bind existing training task, register current monitor/gallery defaults, preserve user presets.
    Key Params: fixed user-supplied artifacts only; no directory scan of mounted drive.
    State/Dependencies: ResourceStore; read failures logged without preventing helper startup.
    """
    task_ref = "task/latest_vosr_v3m_v5_500k_20261102.md"
    sources = [dict(kind="monitor", title="当前训练与数据处理", source="Y:/guang.yang/TextSR/data/current_jobs_monitor_20261030/index.html"),
               dict(kind="gallery", title="Mojito280 · 六模型500K与历史参照", source=str(ROOT / "evaluation/latest_six_500k_mojito280_20261103/manifest.json"),
                    defaults=dict(sample="051", models=["v3m_guard_raw", "v3_input", "v3m_mix50_raw", "previous250_nocp_raw", "v3m_l1", "v5_l1_raw"], baseline="v3m_mix50_raw"))]
    for source in sources:
        if any(item["kind"] == source["kind"] and Path(item["source"]) == Path(source["source"]) for item in resources.list()):
            continue
        try:
            resources.register(dict(source, taskRef=task_ref, projects=["Real-ESRGAN"]), resources.tasks())
        except (OSError, ValueError) as exc:
            print(f"Observation seed unavailable: {source['kind']}: {exc}", flush=True)


def main():
    """@luna-comment
    Input: optional --port integer, default 8765; no external bind option.
    Output: loopback URL and HTTP service until keyboard interrupt.
    Key Steps: parse port, open document store, serve loopback file/workspace actions.
    Key Params: fixed 127.0.0.1 host prevents exposure on network interfaces.
    State/Dependencies: standard library and Handler; no external packages.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.store = Store(ROOT / ".artifacts" / "ai-control-center" / "tasks.sqlite3")
        server.resources = ResourceStore(ROOT / ".artifacts" / "ai-control-center" / "resources.sqlite3")
        seed_observations(server.resources)
        print(f"AI Workspace: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.store.db.close()
            server.resources.db.close()


if __name__ == "__main__":
    main()
