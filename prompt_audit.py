#!/usr/bin/env python3
"""Audit prompt schemas and emit a reproducibility manifest."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PAPER_EXPECTED = {
    "description_career_bias.json": 2700,
    "descriptions_persona_bias.json": 4000,
    "descriptions_status_bias.json": 200,
    "descriptions_two_axis.json": 3200,
    "descriptions_multi_axis.json": 3200,
}
REQUIRED_KEYS = {"id", "description", "trait", "keywords", "prompt_text"}


def audit_file(path):
    raw = path.read_bytes()
    errors = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"file": str(path), "count": 0, "sha256": hashlib.sha256(raw).hexdigest(), "errors": [str(exc)]}
    if not isinstance(data, list):
        errors.append("root must be a list")
        data = []
    ids = []
    for index, item in enumerate(data):
        missing = REQUIRED_KEYS - set(item) if isinstance(item, dict) else REQUIRED_KEYS
        if missing:
            errors.append(f"item {index} missing: {', '.join(sorted(missing))}")
        elif not all(str(item[key]).strip() for key in REQUIRED_KEYS):
            errors.append(f"item {index} contains an empty required value")
        if isinstance(item, dict):
            ids.append(str(item.get("id", "")))
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    return {"file": str(path), "count": len(data), "sha256": hashlib.sha256(raw).hexdigest(), "errors": errors}


def build_manifest(description_dir):
    files = [audit_file(path) for path in sorted(description_dir.glob("*.json"))]
    actual = {Path(item["file"]).name: item["count"] for item in files}
    missing = {name: max(0, expected - actual.get(name, 0)) for name, expected in PAPER_EXPECTED.items()}
    missing = {name: count for name, count in missing.items() if count}
    count_mismatches = {
        name: {"expected": expected, "actual": actual.get(name, 0)}
        for name, expected in PAPER_EXPECTED.items()
        if actual.get(name, 0) != expected
    }
    unexpected = sorted(set(actual) - set(PAPER_EXPECTED))
    schema_valid = all(not item["errors"] for item in files)
    count_alignment = not count_mismatches and not unexpected
    # The paper provides counts but no reference hashes for identity checking.
    # Equal counts alone therefore cannot establish exact prompt reproduction.
    content_identity_verified = False
    return {
        "schema_version": 2,
        "paper_expected_total": sum(PAPER_EXPECTED.values()),
        "repository_total": sum(actual.values()),
        "missing_human_verified_prompts": sum(missing.values()),
        "paper_count_alignment": count_alignment,
        "content_identity_verified": content_identity_verified,
        "exact_paper_reproduction_available": count_alignment and schema_valid and content_identity_verified,
        "missing_by_file": missing,
        "count_mismatches": count_mismatches,
        "unexpected_json_files": unexpected,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit prompt files and write a hash manifest")
    parser.add_argument("--descriptions", default="descriptions")
    parser.add_argument("--output", default="prompt_manifest.json")
    parser.add_argument("--strict-paper", action="store_true", help="Fail when paper prompt counts are unavailable")
    args = parser.parse_args()
    manifest = build_manifest(Path(args.descriptions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    schema_errors = sum(bool(item["errors"]) for item in manifest["files"])
    print(f"Audited {manifest['repository_total']} prompts; missing {manifest['missing_human_verified_prompts']} from paper protocol")
    if schema_errors:
        print(f"[ERROR] {schema_errors} files failed schema validation", file=sys.stderr)
        return 1
    if args.strict_paper and not manifest["exact_paper_reproduction_available"]:
        print("[ERROR] exact paper reproduction is unavailable", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
