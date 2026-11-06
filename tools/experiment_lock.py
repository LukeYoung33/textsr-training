"""Manifest-bound guard for repeatable visual experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    """@luna-comment
    Input: readable file path. Output: SHA256 hex of exact bytes.
    Key Steps: read and hash. Key Params: none. State/Dependencies: read-only filesystem.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    """@luna-comment
    Input: UTF8 JSON path. Output: object, or ValueError for non-object root.
    Key Steps: decode and validate. Key Params: none. State/Dependencies: json/filesystem.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def _manifest_digest(value: dict[str, Any]) -> str:
    """@luna-comment
    Input: manifest mapping. Output: deterministic semantic SHA256.
    Key Steps: sorted compact UTF8 serialization then hash. Key Params: preserve Unicode.
    State/Dependencies: json/hashlib, no mutations.
    """
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(manifest_path: Path) -> dict[str, Any]:
    """@luna-comment
    Input: repository visual-experiment manifest. Output: manifest and authorized-code hashes.
    Key Steps: require designed state/fields, locate root, reject forbidden tokens, hash files.
    Key Params: required fields and forbidden_tokens. State/Dependencies: read-only project files.
    """
    manifest = _load(manifest_path)
    required = ("experiment_id", "status", "authorized_postprocess", "authorized_files", "forbidden_tokens", "output_directory", "required_debug")
    missing = [key for key in required if not manifest.get(key)]
    if manifest.get("status") != "designed" or missing:
        raise ValueError(f"manifest is not executable: status={manifest.get('status')!r}, missing={missing}")
    root = next((parent for parent in manifest_path.parents if (parent / "tools").is_dir()), None)
    if root is None:
        raise ValueError("manifest is not inside a repository containing tools/")
    fingerprints: dict[str, str] = {}
    for name in manifest["authorized_files"]:
        path = root / name
        if not path.is_file():
            raise ValueError(f"authorized file missing: {name}")
        text = path.read_text(encoding="utf-8")
        forbidden = [token for token in manifest["forbidden_tokens"] if token in text]
        if forbidden:
            raise ValueError(f"forbidden token in {name}: {forbidden}")
        fingerprints[name] = _sha256(path)
    return {"manifest_sha256": _manifest_digest(manifest), "code_sha256": fingerprints}


def attest(manifest_path: Path, output: Path) -> None:
    """@luna-comment
    Input: valid manifest and receipt destination. Output: written implementation attestation.
    Key Steps: validate, retain identity, create parent and write JSON.
    Key Params: output path. State/Dependencies: build-owned receipt filesystem.
    """
    receipt = validate(manifest_path)
    manifest = _load(manifest_path)
    receipt.update({"experiment_id": manifest["experiment_id"], "manifest": str(manifest_path), "implementation_attested": True})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_receipt(manifest_path: Path, receipt_path: Path) -> None:
    """@luna-comment
    Input: current manifest and saved attestation. Output: None or mismatch exception.
    Key Steps: recompute and compare manifest/code hashes. Key Params: none.
    State/Dependencies: read-only authorized source files and receipt.
    """
    expected = validate(manifest_path)
    receipt = _load(receipt_path)
    if receipt.get("manifest_sha256") != expected["manifest_sha256"] or receipt.get("code_sha256") != expected["code_sha256"]:
        raise ValueError("receipt does not match the approved manifest or current code")


def main() -> int:
    """@luna-comment
    Input: validate/attest/verify-receipt CLI. Output: result/receipt and success exit code.
    Key Steps: parse subcommand and dispatch. Key Params: explicit manifest/receipt paths.
    State/Dependencies: helpers above; does not itself launch remote experiments.
    """
    parser = argparse.ArgumentParser(description="Block experiments not bound to an approved manifest")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "attest", "verify-receipt"):
        item = sub.add_parser(name)
        item.add_argument("--manifest", type=Path, required=True)
        if name == "attest": item.add_argument("--output", type=Path, required=True)
        if name == "verify-receipt": item.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate": print(json.dumps(validate(args.manifest), ensure_ascii=False, indent=2))
    elif args.command == "attest": attest(args.manifest, args.output)
    else: verify_receipt(args.manifest, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
