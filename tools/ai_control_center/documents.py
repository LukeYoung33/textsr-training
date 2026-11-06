"""Workspace documents and historical records only; no Codex transport or inference."""
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

TEMPLATE = Path(__file__).resolve().parents[2] / "apps" / "ai-control-center" / "TASK_ISSUE_TEMPLATE.md"
CATEGORIES = {"development", "training", "testing", "benchmark", "conversion", "cpp-pipeline", "document-detection", "documentation", "other"}


def markdown_block(value):
    """@luna-comment Input: untrusted text. Output: literal fenced Markdown section.
    Key Steps: choose a longer backtick fence than any input run.
    Key Params: at least three backticks. State/Dependencies: regex; no HTML interpretation.
    """
    text = str(value or "Not recorded")
    fence = "`" * max(3, 1 + max((len(run[0]) for run in re.finditer(r"`+", text)), default=0))
    return fence + "text\n" + text + "\n" + fence


def render_markdown(task):
    """@luna-comment Input: formal task snapshot. Output: filled canonical issuance template.
    Key Steps: single-line encode metadata, fence free text, substitute known placeholders.
    Key Params: no inferred initiator or user acceptance. State/Dependencies: TEMPLATE; no execution.
    """
    metadata = task.get("metadata", {})
    values = dict(task_id=task["id"], title=metadata.get("title", "Legacy task: title not recorded"),
        category=metadata.get("category", "not recorded"), priority=metadata.get("priority", "normal"),
        initiator=metadata.get("initiator", "not recorded"), issued_at_iso8601=task.get("createdAt", "not recorded"),
        started_at_or_not_started=task.get("startedAt", "not started"), ended_at_or_not_ended=task.get("endedAt", "not ended"),
        status=task["status"], model=task.get("model", "not recorded"), workspace_configuration=task.get("workspace", "not recorded"),
        primary_directory=task.get("cwd", "not recorded"), thread_id_or_not_started=task.get("threadId") or "not started",
        turn_id_or_not_started=task.get("turnId") or "not started")
    values = {key: json.dumps(value, ensure_ascii=False) for key, value in values.items()}
    values.update(purpose=markdown_block(metadata.get("purpose")), instruction_verbatim=markdown_block(task.get("instruction")),
        project_names_ids_and_canonical_paths=markdown_block("\n".join(f"{p['name']} [{p['id']}] {p['path']}" for p in task.get("projects", []))),
        deliverables=markdown_block(metadata.get("deliverables")), acceptance_criteria=markdown_block(metadata.get("acceptance")),
        codex_message_texts_with_phase=markdown_block("\n\n".join(f"[{m.get('phase') or 'message'}]\n{m['text']}" for m in task.get("messages", [])) or "No result recorded yet. Continue in the existing VSCode Codex panel."),
        error_or_none=markdown_block(json.dumps(task["error"], ensure_ascii=False) if task.get("error") else "None"),
        record_file=json.dumps(task.get("recordFile", "not recorded"), ensure_ascii=False))
    template = TEMPLATE.read_text(encoding="utf-8")
    return re.sub(r"\{\{([a-z_]+)\}\}", lambda match: values[match[1]], template)


def now():
    """@luna-comment Input: none. Output: UTC timestamp. Key Steps: format clock.
    Key Params: ISO 8601. State/Dependencies: datetime; no shared mutation.
    """
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        """@luna-comment Input: repository database path. Output: ready store.
        Key Steps: create schema, interrupt stale tasks. Key Params: SQLite transactions.
        State/Dependencies: private RLock and connection; never resumes tasks automatically.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.record_dir = path.parent / "issued"
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, data TEXT NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS task_events ON events(task, seq)")
        self.db.commit()
        for task in self.list():
            if task["status"] in ("queued", "running", "reconnecting", "awaiting_approval", "awaiting_input"):
                self.update(task["id"], status="interrupted", error="Local service restarted; task was not resumed.", endedAt=now())

    def update(self, task_id, **fields):
        """@luna-comment Input: task ID and changed fields. Output: merged record.
        Key Steps: locked merge/upsert, atomic formal Markdown replacement, commit or rollback.
        Key Params: conversations never emit issuance files. State/Dependencies: SQLite/TEMPLATE/filesystem.
        """
        with self.lock:
            row = self.db.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()
            task = json.loads(row[0]) if row else {"id": task_id}
            task.update(fields)
            try:
                self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?, ?)", (task_id, json.dumps(task, ensure_ascii=False)))
                if task.get("kind", "task") == "task":
                    self.record_dir.mkdir(parents=True, exist_ok=True)
                    temporary = self.record_dir / (task_id + ".tmp")
                    temporary.write_text(render_markdown(task), encoding="utf-8")
                    temporary.replace(self.record_dir / (task_id + ".md"))
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            return task

    def list(self, kind=None):
        """@luna-comment Input: none. Output: saved task snapshots, newest first.
        Key Steps: query/decode/filter. Key Params: kind selects formal tasks or conversations.
        State/Dependencies: locked SQLite; omitted kind preserves internal recovery across both.
        """
        with self.lock:
            records = [json.loads(row[0]) for row in self.db.execute("SELECT data FROM tasks ORDER BY rowid DESC")]
            return [task for task in records if kind is None or task.get("kind", "task") == kind]

    def get(self, task_id, after=0):
        """@luna-comment Input: known task ID and global event cursor. Output: task plus events.
        Key Steps: lookup, decode ordered batch. Key Params: max 500 events per poll.
        State/Dependencies: locked SQLite; unknown IDs raise ValueError.
        """
        with self.lock:
            row = self.db.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise ValueError("Unknown task")
            task = json.loads(row[0])
            task["events"] = [{"seq": seq, **json.loads(data)} for seq, data in self.db.execute(
                "SELECT seq, data FROM events WHERE task=? AND seq>? ORDER BY seq LIMIT 500", (task_id, after))]
            return task

    def event(self, task_id, method, params):
        """@luna-comment Input: task ID, protocol method and payload. Output: durable ordered event.
        Key Steps: serialize and insert. Key Params: UTC receipt time. State/Dependencies: locked SQLite.
        """
        with self.lock:
            self.db.execute("INSERT INTO events(task, data) VALUES (?, ?)",
                            (task_id, json.dumps(dict(method=method, params=params, at=now()), ensure_ascii=False)))
            self.db.commit()

    def create_task(self, payload, inventory):
        """@luna-comment Input: explicit formal metadata/instruction and current workspace inventory.
        Output: saved Markdown issuance record only; never launches a process or marks execution.
        Key Steps: validate fields/project IDs, freeze scope, write document and DB snapshot.
        Key Params: required title/category/initiator/purpose; current roots only, no guessed identity.
        State/Dependencies: Store/update; filesystem writes stay in the local record directory.
        """
        instruction = payload.get("instruction")
        if not isinstance(instruction,str) or not instruction.strip() or len(instruction)>20000:
            raise ValueError("Provide a task description of 1–20000 characters.")
        supplied=payload.get("metadata",{})
        if not isinstance(supplied,dict):raise ValueError("Formal task metadata is required.")
        metadata={}
        for key in ("title","category","initiator","purpose"):
            value=supplied.get(key)
            if not isinstance(value,str) or not value.strip() or len(value)>(4000 if key=="purpose" else 200):
                raise ValueError(f"Task requires valid {key}.")
            metadata[key]=value.strip()
        if metadata["category"] not in CATEGORIES:raise ValueError("Invalid task category.")
        metadata["priority"]=supplied.get("priority","normal")
        if metadata["priority"] not in ("low","normal","high","urgent"):raise ValueError("Invalid priority.")
        for key in ("deliverables","acceptance"):
            value=supplied.get(key,"")
            if not isinstance(value,str) or len(value)>4000:raise ValueError(f"Invalid {key}.")
            metadata[key]=value.strip()
        ids,primary=payload.get("projects"),payload.get("primary")
        projects={project["id"]:project for project in inventory["projects"]}
        if not isinstance(ids,list) or not ids or any(not isinstance(value,str) for value in ids) or len(set(ids))!=len(ids):
            raise ValueError("Choose one or more distinct workspace projects.")
        if primary not in ids or any(value not in projects or not projects[value]["exists"] for value in ids):
            raise ValueError("Invalid project scope or primary directory; refresh the workspace.")
        task_id=str(uuid.uuid4())
        task=self.update(task_id,kind="task",status="issued",metadata=metadata,instruction=instruction.strip(),
            projects=[dict(projects[value]) for value in ids],primary=primary,cwd=projects[primary]["path"],
            workspace=inventory["workspace"],createdAt=now(),recordFile=str(self.record_dir/(task_id+".md")),
            execution="not_observed",acceptance="pending",messages=[])
        return self.document(task["id"])

    def document(self, task_id):
        """@luna-comment Input: saved task ID. Output: current document, recorded feedback/status and file URI.
        Key Steps: read allowlisted task file, extract only unfenced headings, preserve literal text.
        Key Params: document status is self-reported; never interpreted as observed Codex execution.
        State/Dependencies: SQLite metadata and Markdown file; no session/credential/model access.
        """
        task=self.get(task_id)
        path=self.record_dir/(task_id+".md")
        text=path.read_text(encoding="utf-8") if path.is_file() else "Legacy record: no Markdown document was generated."
        task.update(document=text,documentAvailable=path.is_file(),recordFile=str(path),fileUri="vscode://file/"+quote(path.resolve().as_posix().lstrip("/"),safe="/:"))
        identity=text.split("## Purpose",1)[0]
        match=re.search(r"^- Status:\s*(.+)$",identity,re.MULTILINE)
        if match:
            try:task["fileStatus"]=json.loads(match[1])
            except ValueError:task["fileStatus"]=match[1].strip()
        fence=None;collecting=False;feedback=[]
        for line in text.splitlines():
            marker=re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$",line)
            if marker:
                token=marker[1]
                if fence is None:fence=token
                elif token[0]==fence[0] and len(token)>=len(fence) and not marker[2].strip():fence=None
                if collecting:feedback.append(line)
                continue
            if fence is None and line=="## Feedback / Results":collecting=True;continue
            if fence is None and collecting and line.startswith("## "):break
            if collecting:feedback.append(line)
        task["feedback"]="\n".join(feedback).strip()
        return task
