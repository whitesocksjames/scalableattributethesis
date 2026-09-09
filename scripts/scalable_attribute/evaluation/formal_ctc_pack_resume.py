#!/usr/bin/env python3
"""Build and execute resumable formal CTC point packs.

This module changes orchestration only. Each matrix row is still evaluated by
the existing one-point formal runner in an independent subprocess.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


BASE_ENDPOINT = "Ours Base"
FULL_ENDPOINT = "Ours Full"
ORIGINAL_ENDPOINT = "Original Unicorn"


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _aggregate_path(run_root: Path, row: Mapping[str, str]) -> Path:
    return run_root / row["output_rel"] / "aggregate" / "aggregate.json"


def formal_point_state(run_root: Path, row: Mapping[str, str]) -> dict[str, Any]:
    """Return a fail-closed state for one CTC point's aggregate artifact."""

    path = _aggregate_path(run_root, row)
    if not path.is_file():
        return {"reusable": False, "reason": "aggregate_missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"reusable": False, "reason": f"aggregate_unreadable:{exc}",
                "path": str(path)}
    expected = ([BASE_ENDPOINT, FULL_ENDPOINT]
                if row["method"] == "Ours" else [ORIGINAL_ENDPOINT])
    statuses = payload.get("endpoint_statuses")
    reusable = (
        payload.get("status") == "PASS"
        and payload.get("formal_status") == "FORMAL_REUSABLE"
        and isinstance(statuses, dict)
        and all(statuses.get(name) == "FORMAL_REUSABLE" for name in expected)
    )
    return {
        "reusable": reusable,
        "reason": "formal_reusable" if reusable else "aggregate_not_reusable",
        "path": str(path),
        "status": payload.get("status"),
        "formal_status": payload.get("formal_status"),
        "endpoint_statuses": statuses,
    }


def next_retry_output(run_root: Path, task_id: str) -> str:
    base = Path("results") / "retries" / task_id
    attempt = 1
    while (run_root / base / f"attempt_{attempt:02d}").exists():
        attempt += 1
    return str(base / f"attempt_{attempt:02d}")


def parse_groups(values: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    assignment: dict[str, str] = {}
    for value in values:
        name, separator, sample_values = value.partition("=")
        samples = [sample for sample in sample_values.split(",") if sample]
        if not separator or not name or not samples:
            raise ValueError(f"invalid --group value: {value!r}")
        if name not in order:
            order.append(name)
        for sample in samples:
            if sample in assignment:
                raise ValueError(f"sample assigned more than once: {sample}")
            assignment[sample] = name
    return order, assignment


def build_remaining(args: argparse.Namespace) -> int:
    fields, source_rows = read_tsv(Path(args.full_matrix))
    group_order, assignment = parse_groups(args.group)
    run_root = Path(args.run_root)
    remaining: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    for source in source_rows:
        state = formal_point_state(run_root, source)
        if state["reusable"]:
            skipped.append({"task_id": source["task_id"], "state": state})
            continue
        sample = source["sample_id"]
        if sample not in assignment:
            raise ValueError(f"remaining sample has no pack assignment: {sample}")
        row = dict(source)
        output = run_root / row["output_rel"]
        if output.exists():
            row["output_rel"] = next_retry_output(run_root, row["task_id"])
        row["launch_status"] = args.launch_status
        row["reason"] = "resumable_remaining_pack"
        row["pack_group"] = assignment[sample]
        remaining.append(row)
    point_fields = fields + (["pack_group"] if "pack_group" not in fields else [])
    write_tsv(Path(args.output_points), point_fields, remaining)
    packs = []
    for group in group_order:
        indices = [index for index, row in enumerate(remaining)
                   if row["pack_group"] == group]
        if not indices:
            continue
        packs.append({
            "pack_id": group,
            "point_indices": ",".join(map(str, indices)),
            "point_task_ids": ",".join(remaining[index]["task_id"] for index in indices),
            "samples": ",".join(dict.fromkeys(remaining[index]["sample_id"] for index in indices)),
        })
    write_tsv(Path(args.output_packs),
              ["pack_id", "samples", "point_indices", "point_task_ids"], packs)
    atomic_json(Path(args.output_summary), {
        "status": "PASS", "source_points": len(source_rows),
        "skipped_reusable": len(skipped), "remaining_points": len(remaining),
        "packs": len(packs), "skipped": skipped,
    })
    print(json.dumps({"status": "PASS", "remaining": len(remaining),
                      "packs": len(packs)}, sort_keys=True))
    return 0


def _next_pack_attempt(evidence_root: Path, pack_id: str) -> Path:
    base = evidence_root / pack_id
    attempt = 1
    while (base / f"attempt_{attempt:02d}").exists():
        attempt += 1
    return base / f"attempt_{attempt:02d}"


def run_pack(args: argparse.Namespace) -> int:
    _, packs = read_tsv(Path(args.pack_manifest))
    _, points = read_tsv(Path(args.point_matrix))
    if args.pack_index < 0 or args.pack_index >= len(packs):
        raise IndexError("pack index outside manifest")
    pack = packs[args.pack_index]
    run_root = Path(args.run_root)
    attempt_dir = _next_pack_attempt(Path(args.evidence_root), pack["pack_id"])
    attempt_dir.mkdir(parents=True)
    summary: dict[str, Any] = {
        "pack": pack, "attempt_dir": str(attempt_dir),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "points": [],
    }
    failures = 0
    for index in [int(value) for value in pack["point_indices"].split(",")]:
        point = points[index]
        before = formal_point_state(run_root, point)
        if before["reusable"]:
            record = {"task_id": point["task_id"],
                      "action": "SKIP_FORMAL_REUSABLE", "success": True,
                      "state": before}
            summary["points"].append(record)
            atomic_json(attempt_dir / "point_status" / f"{point['task_id']}.json",
                        record)
            atomic_json(attempt_dir / "pack_progress.json", summary)
            continue
        command = [sys.executable, args.point_runner, "--matrix", args.point_matrix,
                   "--index", str(index)]
        started = dt.datetime.now(dt.timezone.utc)
        launch_error = None
        try:
            returncode = subprocess.run(command, check=False).returncode
        except OSError as exc:
            returncode = None
            launch_error = f"{type(exc).__name__}: {exc}"
        ended = dt.datetime.now(dt.timezone.utc)
        after = formal_point_state(run_root, point)
        success = returncode == 0 and after["reusable"]
        record = {
            "task_id": point["task_id"], "action": "RUN", "command": command,
            "returncode": returncode, "launch_error": launch_error,
            "success": success,
            "started_utc": started.isoformat(), "ended_utc": ended.isoformat(),
            "elapsed_seconds": (ended - started).total_seconds(), "state": after,
        }
        summary["points"].append(record)
        atomic_json(attempt_dir / "point_status" / f"{point['task_id']}.json",
                    record)
        failures += int(not success)
        atomic_json(attempt_dir / "pack_progress.json", summary)
    summary["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    summary["failure_count"] = failures
    summary["status"] = "PASS" if failures == 0 else "PARTIAL"
    atomic_json(attempt_dir / "pack_result.json", summary)
    return int(failures != 0)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--full-matrix", required=True)
    build.add_argument("--run-root", required=True)
    build.add_argument("--group", action="append", required=True)
    build.add_argument("--launch-status", required=True)
    build.add_argument("--output-points", required=True)
    build.add_argument("--output-packs", required=True)
    build.add_argument("--output-summary", required=True)
    build.set_defaults(function=build_remaining)
    run = commands.add_parser("run")
    run.add_argument("--pack-manifest", required=True)
    run.add_argument("--pack-index", required=True, type=int)
    run.add_argument("--point-matrix", required=True)
    run.add_argument("--point-runner", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--evidence-root", required=True)
    run.set_defaults(function=run_pack)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
