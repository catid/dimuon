#!/usr/bin/env python3
"""Predeclared, sequential eight-GPU optimizer study; standard library only.

Plans by default. Pass --execute to launch jobs. See STUDY_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
METHODS = ("adamw", "muon", "dimuon")
INITIALIZATIONS = ("lora", "nora", "inilora_alpha")
DATASETS = ("gsm8k", "math")
SCHEMA = 1
DEFAULT_GRID = {
    init: {
        "adamw": [3e-5, 1e-4, 3e-4],
        "muon": [3e-4, 1e-3, 3e-3],
        "dimuon": [3e-3, 1.2e-2, 4.8e-2] if init == "lora" else [1e-3, 3e-3, 1.2e-2],
    }
    for init in INITIALIZATIONS
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def csv_values(value, allowed=None, integer=False):
    values = value.split(",")
    if integer:
        values = [int(v) for v in values]
    if not values or len(set(values)) != len(values) or any(v == "" for v in values):
        raise ValueError("comma-separated lists must be nonempty and unique")
    if allowed and not set(values) <= set(allowed):
        raise ValueError(f"expected a subset of {allowed}, got {values}")
    return values


def manifest_identity(path):
    """Validate records and use question hashes to check split separation."""
    path = Path(path).resolve(strict=True)
    questions = set()
    count = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{line_number}: missing question")
            if not isinstance(record.get("gold"), str):
                raise ValueError(f"{path}:{line_number}: gold must be a string")
            normalized = " ".join(question.split()).casefold()
            questions.add(hashlib.sha256(normalized.encode()).hexdigest())
            count += 1
    if not count:
        raise ValueError(f"empty manifest: {path}")
    return {"path": str(path), "sha256": file_hash(path), "records": count}, questions


def model_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("--model must be a local checkpoint directory containing config.json")
    # Avoid reading tens/hundreds of GB of weight shards at every resume. Small
    # config/tokenizer files are hashed; large files are identified by metadata.
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or ".cache" in item.relative_to(path).parts:
            continue
        stat = item.stat()
        info = {"path": str(item.relative_to(path)), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= 16 * 1024 * 1024:
            info["sha256"] = file_hash(item)
        files.append(info)
    return {"path": str(path), "files": files, "large_weight_hashes": False}


def source_identity():
    return {name: file_hash(ROOT / name) for name in
            ("run_study.py", "dimuon_lora_benchmark.py", "dimuon.py")}


def environment_identity(python):
    script = """import importlib.metadata as m, json, platform, sys
versions = {}
for name in ('torch', 'transformers', 'numpy', 'tokenizers', 'safetensors'):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': sys.version, 'executable': sys.executable,
                  'platform': platform.platform(), 'packages': versions}))
"""
    return json.loads(subprocess.check_output([python, "-c", script], text=True))


def prepare_plan(args):
    datasets = csv_values(args.datasets, DATASETS)
    inits = csv_values(args.initializations, INITIALIZATIONS)
    tune = csv_values(args.tune_seeds, integer=True)
    confirm = csv_values(args.confirm_seeds, integer=True)
    gpus = csv_values(args.gpus)
    if len(gpus) != 8:
        raise ValueError("this study requires exactly eight distinct --gpus")
    if set(tune) & set(confirm) or min(tune + confirm) < 0:
        raise ValueError("tuning and confirmation seeds must be nonnegative and disjoint")
    for name in ("steps", "batch", "seq", "rank", "last_n", "max_rows",
                 "val_steps", "test_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.warmup < args.steps or args.alpha <= 0 or args.timeout_seconds < 0:
        raise ValueError("require 0 <= warmup < steps, alpha > 0, timeout >= 0")
    grid = read_json(args.lr_grid) if args.lr_grid else DEFAULT_GRID
    selected_grid = {}
    for init in inits:
        selected_grid[init] = {}
        for method in METHODS:
            values = grid[init][method]
            if not isinstance(values, list) or not values or len(set(values)) != len(values):
                raise ValueError(f"{init}/{method}: LR candidates must be a nonempty unique list")
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                   not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{init}/{method}: LR candidates must be positive finite numbers")
            selected_grid[init][method] = sorted(values)
    sizes = {len(values) for arms in selected_grid.values() for values in arms.values()}
    if len(sizes) != 1:
        raise ValueError("every optimizer/initialization must receive the same number of LR candidates")
    manifests, question_sets = {}, {}
    for dataset in datasets:
        manifests[dataset] = {}
        for split in ("train", "val", "test"):
            path = getattr(args, f"{dataset}_{split}")
            if not path:
                raise ValueError(f"--{dataset}-{split} is required")
            info, questions = manifest_identity(path)
            manifests[dataset][split] = info
            question_sets[(dataset, split)] = questions
    # Reject identical/overlapping questions across any differently named
    # splits, including cross-dataset train/test contamination.
    for i, (key_a, questions_a) in enumerate(question_sets.items()):
        for key_b, questions_b in list(question_sets.items())[i + 1:]:
            if key_a != key_b and questions_a & questions_b:
                raise ValueError(f"question overlap between {key_a} and {key_b}; create disjoint manifests")
    # Preserve the venv entry-point path.  ``Path.resolve()`` follows a venv's
    # symlink into the base interpreter and silently drops the venv's torch and
    # other packages.  Absolute normalization without symlink resolution keeps
    # the selected environment intact while still making the plan portable.
    python_path = Path(args.python).absolute()
    if not python_path.is_file():
        raise ValueError(f"--python is not a file: {python_path}")
    python = str(python_path)
    plan = {
        "schema": SCHEMA, "datasets": datasets, "initializations": inits,
        "methods": list(METHODS), "lr_grid": selected_grid,
        "tune_seeds": tune, "confirm_seeds": confirm,
        "model": model_identity(args.model), "manifests": manifests,
        "source_sha256": source_identity(), "environment": environment_identity(python),
        "python": python, "gpus": gpus, "world": 8,
        "training": {key: getattr(args, key) for key in
                     ("steps", "warmup", "batch", "seq", "rank", "alpha", "last_n", "max_rows")},
        "val_steps": args.val_steps, "test_steps": args.test_steps,
        "timeout_seconds": args.timeout_seconds,
        "selection": "minimum mean validation CE over all tuning seeds; exact ties choose smaller LR",
        "budget": {"tuning_jobs": len(datasets) * len(inits) * 3 * sizes.pop() * len(tune),
                   "confirmation_jobs": len(datasets) * len(inits) * 3 * len(confirm)},
    }
    return plan


def jobs_for(plan, stage, selection=None):
    jobs = []
    for dataset_index, dataset in enumerate(plan["datasets"]):
        for init_index, init in enumerate(plan["initializations"]):
            seeds = plan["tune_seeds"] if stage == "tune" else plan["confirm_seeds"]
            count = len(plan["lr_grid"][init]["adamw"]) if stage == "tune" else 1
            for candidate in range(count):
                for seed_index, seed in enumerate(seeds):
                    offset = (dataset_index + init_index + candidate + seed_index) % len(METHODS)
                    for method in METHODS[offset:] + METHODS[:offset]:
                        cell = f"{dataset}/{init}/{method}"
                        if stage == "tune":
                            lr = plan["lr_grid"][init][method][candidate]
                        else:
                            winner = selection["cells"][cell]["winner_lr"]
                            if winner is None:
                                continue
                            lr = winner
                        jobs.append({"stage": stage, "dataset": dataset, "init": init,
                                     "method": method, "lr": lr, "seed": seed})
    return jobs


def expected_args(plan, job, output):
    split = "val" if job["stage"] == "tune" else "test"
    result = dict(plan["training"])
    result.update(model=plan["model"]["path"], method=job["method"], init=job["init"],
                  seed=job["seed"], grouped_dimuon=True, out=str(output),
                  manifest=plan["manifests"][job["dataset"]]["train"]["path"],
                  eval_manifest=plan["manifests"][job["dataset"]][split]["path"],
                  eval_steps=plan["val_steps"] if split == "val" else plan["test_steps"])
    result[f"{job['method']}_lr"] = job["lr"]
    return result


def command_for(plan, job, output):
    cmd = [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
           str(ROOT / "dimuon_lora_benchmark.py")]
    for key, value in expected_args(plan, job, output).items():
        cmd.append("--" + key.replace("_", "-"))
        if value is not True:
            cmd.append(str(value))
    return cmd


def validate_output(path, plan, job):
    payload = read_json(path)
    expected = expected_args(plan, job, path)
    for key, value in expected.items():
        if payload.get("args", {}).get(key) != value:
            raise ValueError(f"result args mismatch for {key}")
    rows = payload.get("results", [])
    if len(rows) != 1:
        raise ValueError("expected exactly one optimizer result")
    row = rows[0]
    for key in ("method", "steps", "warmup", "model", "rank", "alpha", "last_n"):
        if row.get(key) != expected[key]:
            raise ValueError(f"result metadata mismatch for {key}")
    if row.get("world") != 8 or row.get("lr") != job["lr"]:
        raise ValueError("result world/LR mismatch")
    losses = row.get("losses", [])
    if len(losses) != plan["training"]["steps"] or row.get("nonfinite_steps") != []:
        raise ValueError("incomplete/nonfinite training result")
    for key in ("eval_loss", "first_loss", "final_loss", "mean_step_ms", "tokens_per_s"):
        value = row.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {key}")
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in losses):
        raise ValueError("invalid training loss sequence")
    for key in ("trainable_params", "adapter_matrices", "optimizer_state_bytes"):
        if not isinstance(row.get(key), int) or row[key] <= 0:
            raise ValueError(f"invalid {key}")
    return row


def job_directory(out, job):
    label = "_".join(str(job[key]) for key in ("stage", "dataset", "init", "method", "seed"))
    return out / "jobs" / f"{label}_{digest(job)[:12]}"


def cached_result(out, plan, job):
    directory = job_directory(out, job)
    for receipt_path in sorted(directory.glob("attempt_*/receipt.json"), reverse=True):
        try:
            receipt = read_json(receipt_path)
            if receipt.get("plan_sha256") != digest(plan) or receipt.get("job") != job:
                continue
            if receipt.get("status") == "failed":
                return receipt
            if receipt.get("status") != "success":
                continue
            output = receipt_path.parent / "result.json"
            if receipt.get("output_sha256") != file_hash(output):
                continue
            row = validate_output(output, plan, job)
            if receipt.get("result") != row:
                continue
            return receipt
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def assert_inputs_unchanged(plan):
    if source_identity() != plan["source_sha256"]:
        raise RuntimeError("source changed after preregistration; start a new output directory")
    if environment_identity(plan["python"]) != plan["environment"]:
        raise RuntimeError("Python/package environment changed after preregistration")
    for splits in plan["manifests"].values():
        for item in splits.values():
            if file_hash(item["path"]) != item["sha256"]:
                raise RuntimeError(f"manifest changed after preregistration: {item['path']}")
    if model_identity(plan["model"]["path"]) != plan["model"]:
        raise RuntimeError("checkpoint identity changed after preregistration")


def run_job(out, plan, job):
    cached = cached_result(out, plan, job)
    if cached:
        print(f"RESUME {cached['status']} {job}", flush=True)
        return cached
    assert_inputs_unchanged(plan)
    directory = job_directory(out, job)
    directory.mkdir(parents=True, exist_ok=True)
    indices = [int(p.name.split("_")[1]) for p in directory.glob("attempt_*") if p.is_dir()]
    attempt = directory / f"attempt_{max(indices, default=0) + 1:04d}"
    attempt.mkdir()
    output = attempt / "result.json"
    command = command_for(plan, job, output)
    receipt = {"plan_sha256": digest(plan), "job": job, "command": command,
               "started": now(), "status": "running"}
    write_json(attempt / "receipt.json", receipt)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(plan["gpus"]),
               PYTHONHASHSEED=str(job["seed"]), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    print(f"RUN {job}; log={attempt / 'stdout.log'}", flush=True)
    process = None
    try:
        with (attempt / "stdout.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=plan["timeout_seconds"] or None)
        receipt["returncode"] = code
        if code:
            raise ValueError(f"launcher exit code {code}")
        receipt["result"] = validate_output(output, plan, job)
        receipt["output_sha256"] = file_hash(output)
        receipt["status"] = "success"
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(f"FAILED {job}: {error}", flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        receipt["finished"] = now()
        write_json(attempt / "receipt.json", receipt)
    return receipt


def select_winners(out, plan):
    candidates = collections.defaultdict(lambda: collections.defaultdict(list))
    for job in jobs_for(plan, "tune"):
        receipt = cached_result(out, plan, job)
        if receipt is None:
            raise RuntimeError("all tuning jobs must finish before freezing LR selection")
        cell = "/".join(job[key] for key in ("dataset", "init", "method"))
        candidates[cell][job["lr"]].append(receipt)
    cells = {}
    for cell, lr_results in candidates.items():
        summary = []
        for lr, receipts in sorted(lr_results.items()):
            complete = all(item["status"] == "success" for item in receipts)
            values = [item["result"]["eval_loss"] for item in receipts if item["status"] == "success"]
            summary.append({"lr": lr, "eligible": complete, "successful_seeds": len(values),
                            "mean_val_ce": statistics.mean(values) if complete else None,
                            "receipt_sha256": [digest(item) for item in receipts]})
        eligible = [item for item in summary if item["eligible"]]
        winner = min(eligible, key=lambda item: (item["mean_val_ce"], item["lr"])) if eligible else None
        cells[cell] = {"winner_lr": winner["lr"] if winner else None, "candidates": summary}
    selection = {"plan_sha256": digest(plan), "cells": cells}
    selection_path = out / "selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise RuntimeError("frozen selection no longer matches validated tuning results; refusing to change it")
    write_json(selection_path, selection)
    return selection


def summarize(out, plan, selection):
    summary = {"plan_sha256": digest(plan), "updated": now(), "selection": selection, "cells": {}}
    grouped = collections.defaultdict(list)
    for job in jobs_for(plan, "confirm", selection):
        cell = "/".join(job[key] for key in ("dataset", "init", "method"))
        receipt = cached_result(out, plan, job)
        grouped[cell].append({"seed": job["seed"], "status": receipt["status"] if receipt else "pending",
                              "result": receipt.get("result") if receipt else None})
    for cell, chosen in selection["cells"].items():
        rows = grouped[cell]
        values = [item["result"]["eval_loss"] for item in rows if item["status"] == "success"]
        summary["cells"][cell] = {
            "selected_lr": chosen["winner_lr"], "confirmation": rows,
            "complete": len(values) == len(plan["confirm_seeds"]),
            "successful_seeds": len(values), "planned_seeds": len(plan["confirm_seeds"]),
            "mean_test_ce": statistics.mean(values) if values else None,
            "sample_sd_test_ce": statistics.stdev(values) if len(values) > 1 else None,
        }
    write_json(out / "summary.json", summary)
    return summary


@contextlib.contextmanager
def study_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another study holds {path}; this launcher will not share the GPUs") from None
        yield


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable, help="Python executable in the training environment")
    for dataset in DATASETS:
        for split in ("train", "val", "test"):
            ap.add_argument(f"--{dataset}-{split}")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--initializations", default=",".join(INITIALIZATIONS))
    ap.add_argument("--lr-grid", help="JSON: initialization -> optimizer -> equal-length LR lists")
    ap.add_argument("--tune-seeds", default="1234")
    ap.add_argument("--confirm-seeds", default="2345,3456,4567")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    for name, default in (("steps", 200), ("warmup", 10), ("batch", 2), ("seq", 512),
                          ("rank", 32), ("last-n", 64), ("max-rows", 4096),
                          ("val-steps", 8), ("test-steps", 32)):
        ap.add_argument(f"--{name}", type=int, default=default)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--timeout-seconds", type=float, default=0, help="per-job timeout; 0 disables")
    ap.add_argument("--phase", choices=("all", "tune", "confirm"), default="all")
    ap.add_argument("--execute", action="store_true", help="launch jobs; omitted means plan only")
    ap.add_argument("--lock-file", default=str(Path.home() / ".cache/dimuon/study.lock"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir).resolve()
    # Lock even plan creation so competing processes cannot preregister two
    # incompatible protocols into one directory.
    with study_lock(Path(args.lock_file).resolve()):
        plan = prepare_plan(args)
        plan_path = out / "plan.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise RuntimeError("output directory contains a different code/config/input plan; use a new --out-dir")
        write_json(plan_path, plan)
        print(f"Plan {digest(plan)}: {plan['budget']}; {plan['training']['steps']} updates/job", flush=True)
        if not args.execute:
            print(f"Plan saved to {plan_path}. No jobs launched. Add --execute to run.")
            return 0
        if args.phase in ("all", "tune"):
            for job in jobs_for(plan, "tune"):
                run_job(out, plan, job)
        selection = select_winners(out, plan)
        summary = summarize(out, plan, selection)
        if args.phase in ("all", "confirm"):
            for job in jobs_for(plan, "confirm", selection):
                run_job(out, plan, job)
                summary = summarize(out, plan, selection)
        failed_cells = [cell for cell, value in selection["cells"].items() if value["winner_lr"] is None]
        incomplete = [cell for cell, value in summary["cells"].items() if not value["complete"]]
        print(f"Summary: {out / 'summary.json'}; no valid LR: {len(failed_cells)}; incomplete confirmation: {len(incomplete)}")
        return 1 if failed_cells or (args.phase != "tune" and incomplete) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"study error: {error}", file=sys.stderr)
        sys.exit(2)
