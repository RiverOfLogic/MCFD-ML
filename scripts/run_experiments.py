import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

METHOD_SCRIPTS = {
    "ERM": "ERM.py",
    "DANN": "DANN0.py",
    "M-DANN": "DANN.py",
    "CDAN": "CDAN.py",
    "MCD": "MCD.py",
    "MLDG": "MLDG.py",
    "CDDG": "CDDG.py",
    "MCFD-ML": "MEDG.py",
    "MCFD-ML-no-meta": "MEDG.py",
    "MCFD-ML-no-adv": "MEDG.py",
    "MCFD-ML-no-coral": "MEDG.py",
    "MCFD-ML-no-domain": "MEDG.py",
    "MCFD-ML-no-HSIC": "MEDG.py",
    "MCFD-ML-no-rec": "MEDG.py",
    "MEDG": "MEDG.py",
    "MEDG-no-meta": "MEDG.py",
    "MEDG-no-adv": "MEDG.py",
    "MEDG-no-coral": "MEDG.py",
    "MEDG-no-domain": "MEDG.py",
    "MEDG-no-HSIC": "MEDG.py",
    "MEDG-no-rec": "MEDG.py",
}

MCFD_ML_ABLATIONS = {
    "MCFD-ML": "none",
    "MCFD-ML-no-meta": "no_meta_supervised",
    "MCFD-ML-no-adv": "no_adv",
    "MCFD-ML-no-coral": "no_coral",
    "MCFD-ML-no-domain": "no_domain_supervision",
    "MCFD-ML-no-HSIC": "no_HSIC",
    "MCFD-ML-no-rec": "no_rec",
    "MEDG": "none",
    "MEDG-no-meta": "no_meta_supervised",
    "MEDG-no-adv": "no_adv",
    "MEDG-no-coral": "no_coral",
    "MEDG-no-domain": "no_domain_supervision",
    "MEDG-no-HSIC": "no_HSIC",
    "MEDG-no-rec": "no_rec",
}

# Backward-compatible name used by older local scripts.
MEDG_ABLATIONS = MCFD_ML_ABLATIONS


def base_method(method: str):
    return "MCFD-ML" if method in MCFD_ML_ABLATIONS else method


def method_param_keys(method: str):
    if method in MCFD_ML_ABLATIONS:
        return [method, "MCFD-ML", "MEDG"]
    return [method, base_method(method)]

RAW_FIELDS = [
    "method",
    "dataset",
    "task",
    "repeat",
    "seed",
    "gpu",
    "status",
    "acc",
    "macro_f1",
    "weighted_f1",
    "loss",
    "log_path",
    "model_path",
    "start_time",
    "end_time",
    "duration_sec",
    "return_code",
]

REQUIRED_DATA_FILES = [
    "train_x.npy",
    "train_y.npy",
    "train_info.npy",
    "val_x.npy",
    "val_y.npy",
    "val_info.npy",
    "test_x.npy",
    "test_y.npy",
    "test_info.npy",
]

SUMMARY_FIELDS = [
    "method",
    "dataset",
    "task",
    "n_success",
    "acc_mean",
    "acc_std",
    "weighted_f1_mean",
    "weighted_f1_std",
    "macro_f1_mean",
    "macro_f1_std",
    "loss_mean",
    "loss_std",
]


@dataclass
class Job:
    method: str
    dataset: str
    task: int
    repeat: int
    seed: int
    channels: int
    data_dir: Path
    runtime_config: dict


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_gpus():
    try:
        import torch

        count = torch.cuda.device_count()
        return list(range(count))
    except Exception:
        return []


def resolve_gpus(config):
    gpus = config.get("gpus", "auto")
    if gpus == "auto":
        detected = detect_gpus()
        return detected if detected else [None]
    if gpus in (None, [], "cpu"):
        return [None]
    return [int(g) for g in gpus]


def scheduler_config(config):
    settings = dict(config.get("gpu_scheduler", {}) or {})
    max_jobs = config.get("max_jobs_per_gpu", 1)
    if str(max_jobs).lower() == "auto":
        settings["mode"] = settings.get("mode", "memory")
    else:
        settings.setdefault("mode", "fixed")
    settings.setdefault("poll_interval_sec", 5)
    settings.setdefault("reserve_mb", 1024)
    settings.setdefault("min_free_mb", 512)
    settings.setdefault("startup_grace_sec", 30)
    settings.setdefault("default_job_memory_mb", 3000)
    settings.setdefault("max_jobs_per_gpu_cap", 4)
    settings.setdefault("method_memory_mb", {})
    return settings


def nvidia_smi_memory(gpus):
    physical_gpus = [gpu for gpu in gpus if gpu is not None]
    if not physical_gpus:
        return {}
    cmd = [
        "nvidia-smi",
        f"--id={','.join(str(gpu) for gpu in physical_gpus)}",
        "--query-gpu=index,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return {}

    memory = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpu = int(parts[0])
            memory[gpu] = {"free_mb": int(parts[1]), "total_mb": int(parts[2])}
        except ValueError:
            continue
    return memory


def job_memory_mb(config, job: Job):
    explicit = job.runtime_config.get("_estimated_gpu_memory_mb")
    if explicit:
        return int(explicit)
    settings = scheduler_config(config)
    method_memory = settings.get("method_memory_mb", {})
    for key in method_param_keys(job.method):
        if key in method_memory:
            return int(method_memory[key])
    return int(settings["default_job_memory_mb"])


def running_on_gpu(running, gpu):
    return sum(1 for active in running if active["gpu"] == gpu)


def young_reserved_mb(running, gpu, grace_sec):
    now = datetime.now()
    total = 0
    for active in running:
        if active["gpu"] != gpu:
            continue
        age = (now - active["start_time"]).total_seconds()
        if age < grace_sec:
            total += int(active.get("estimated_gpu_memory_mb", 0))
    return total


def choose_memory_gpu(config, job: Job, gpus, running, memory_snapshot):
    settings = scheduler_config(config)
    required_mb = job_memory_mb(config, job)
    cap = int(settings["max_jobs_per_gpu_cap"])
    reserve_mb = int(settings["reserve_mb"])
    min_free_mb = int(settings["min_free_mb"])
    grace_sec = int(settings["startup_grace_sec"])

    candidates = []
    for gpu in gpus:
        if gpu is None:
            return None
        if running_on_gpu(running, gpu) >= cap:
            continue
        stats = memory_snapshot.get(gpu)
        if not stats:
            continue
        available_mb = stats["free_mb"] - reserve_mb - young_reserved_mb(running, gpu, grace_sec)
        if available_mb >= required_mb and available_mb - required_mb >= min_free_mb:
            candidates.append((available_mb, gpu))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def fixed_slots(config, gpus):
    configured = config.get("max_jobs_per_gpu", 1)
    max_jobs_per_gpu = 1 if str(configured).lower() == "auto" else int(configured)
    return [gpu for gpu in gpus for _ in range(max_jobs_per_gpu)]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def dataset_for_task(config, task: int):
    for dataset_name, dataset_config in config["datasets"].items():
        if task in dataset_config["tasks"]:
            data_root = config.get("data_root", "data")
            data_dir = dataset_config.get("path", Path(data_root) / dataset_name)
            return dataset_name, int(dataset_config["channels"]), resolve_path(data_dir)
    raise ValueError(f"Task {task} is not assigned to any dataset in YAML.")


def validate_data_dirs(jobs):
    checked = {}
    for job in jobs:
        if job.dataset in checked:
            continue
        missing = [name for name in REQUIRED_DATA_FILES if not (job.data_dir / name).exists()]
        checked[job.dataset] = (job.data_dir, missing)

    errors = []
    for dataset, (data_dir, missing) in checked.items():
        if missing:
            errors.append(f"{dataset}: {data_dir} missing {', '.join(missing)}")
    if errors:
        joined = "\n  - ".join(errors)
        raise FileNotFoundError(
            "Dataset paths in YAML are not ready. Please fix datasets.*.path.\n"
            f"  - {joined}"
        )


def method_params(config, method: str, dataset: str, task: int):
    all_params = config.get("params", {})
    params = {}
    for key in method_param_keys(method):
        if key in all_params:
            params = all_params[key]
            break
    if dataset in params:
        params = params[dataset]
    if str(task) in params:
        params = params[str(task)]
    elif task in params:
        params = params[task]
    return dict(params)


def runtime_overrides(config, method: str, dataset: str, task: int, channels: int, params: dict):
    defaults = config.get("defaults", {})
    num_classes = int(defaults.get("num_classes", 7))
    method_base = base_method(method)
    overrides = {
        "_method": method,
        "dataset": dataset,
        "TASK": task,
        "channels": channels,
        "num_classes": num_classes,
        "num_workers": int(params.get("num_workers", defaults.get("num_workers", 0))),
        "pin_memory": bool(params.get("pin_memory", defaults.get("pin_memory", True))),
        "persistent_workers": bool(params.get("persistent_workers", defaults.get("persistent_workers", True))),
        "prefetch_factor": int(params.get("prefetch_factor", defaults.get("prefetch_factor", 2))),
    }

    if method_base == "ERM":
        overrides.update(
            ERM_num_classes=num_classes,
            ERM_epochs=params.get("epochs", 100),
            ERM_batch_size=params.get("batch_size", 128),
            ERM_lr=params.get("lr", 0.0005),
        )
    elif method_base == "DANN":
        overrides.update(
            DANN0_num_classes=num_classes,
            DANN0_epochs=params.get("epochs", 100),
            DANN0_batch_size=params.get("batch_size", 128),
            DANN0_lr=params.get("lr", 0.0005),
            DANN0_weight_domain=params.get("weight_domain", 0.5),
        )
    elif method_base == "M-DANN":
        overrides.update(
            DANN_num_classes=num_classes,
            DANN_epochs=params.get("epochs", 100),
            DANN_batch_size=params.get("batch_size", 128),
            DANN_lr=params.get("lr", 0.0005),
            DANN_weight_domain=params.get("weight_domain", 1.0),
        )
    elif method_base == "CDAN":
        overrides.update(
            CDAN_num_classes=num_classes,
            CDAN_epochs=params.get("epochs", 100),
            CDAN_batch_size=params.get("batch_size", 64),
            CDAN_lr=params.get("lr", 0.0005),
            CDAN_trade_off=params.get("trade_off", 1.5),
            CDAN_entropy=params.get("entropy", True),
        )
    elif method_base == "MCD":
        overrides.update(
            MCD_num_classes=num_classes,
            MCD_epochs=params.get("epochs", 100),
            MCD_batch_size=params.get("batch_size", 128),
            MCD_lr=params.get("lr", 0.0005),
        )
    elif method_base == "MLDG":
        overrides.update(
            epochs=params.get("epochs", 100),
            batch_size=params.get("batch_size", 64),
            lr=params.get("lr", 0.0001),
            MLDG_inner_lr=params.get("inner_lr", 0.001),
            MLDG_beta=params.get("beta", 1.0),
        )
    elif method_base == "CDDG":
        overrides.update(
            CDDG_epochs=params.get("epochs", 100),
            CDDG_batch_size=params.get("batch_size", 64),
            CDDG_lr=params.get("lr", 0.0001),
        )
    elif method_base == "MCFD-ML":
        overrides.update(
            epochs=params.get("epochs", 100),
            batch_size=params.get("batch_size", 64),
            lr=params.get("lr", 0.0001),
            weight_outer=params.get("weight_outer", 0.5),
            weight_coral=params.get("weight_coral", 0.3),
            weight_adv=params.get("weight_adv", 1.0),
            weight_domainacc=params.get("weight_domainacc", 0.2),
            weight_HSIC=params.get("weight_HSIC", 0.1),
            weight_rec=params.get("weight_rec", 0.2),
            lr_decay_enabled=params.get("lr_decay_enabled", False),
            lr_decay_step_size=params.get("lr_decay_step_size", 30),
            lr_decay_gamma=params.get("lr_decay_gamma", 0.5),
            medg_ablation=MCFD_ML_ABLATIONS.get(method, "none"),
            medg_method_name=method,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if "gpu_memory_mb" in params:
        overrides["_estimated_gpu_memory_mb"] = int(params["gpu_memory_mb"])

    return overrides


def expand_jobs(config):
    methods = config.get("methods", [])
    tasks = [int(t) for t in config.get("tasks", [])]
    repeats = int(config.get("repeats", 1))
    explicit_seeds = config.get("seeds") or []
    base_seed = int(config.get("base_seed", 42))
    jobs = []

    for method in methods:
        if method not in METHOD_SCRIPTS:
            raise ValueError(f"Method {method} has no script mapping.")
        for task in tasks:
            dataset, channels, data_dir = dataset_for_task(config, task)
            params = method_params(config, method, dataset, task)
            for repeat in range(repeats):
                seed = int(explicit_seeds[repeat]) if repeat < len(explicit_seeds) else base_seed + repeat
                runtime_config = runtime_overrides(config, method, dataset, task, channels, params)
                runtime_config["DIRG_DATA_DIR"] = str(data_dir)
                jobs.append(Job(method, dataset, task, repeat, seed, channels, data_dir, runtime_config))
    return jobs


def make_output_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    template = config.get("output_dir", "experiments/results/{timestamp}")
    out_dir = ROOT / template.format(timestamp=timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "runtime_configs").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    return out_dir


def load_completed(raw_csv: Path):
    completed = set()
    if not raw_csv.exists():
        return completed
    with open(raw_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "success":
                completed.add((row["method"], int(row["task"]), int(row["seed"])))
    return completed


def ensure_raw_csv(path: Path):
    if path.exists():
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RAW_FIELDS).writeheader()


def append_raw(path: Path, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writerow({field: row.get(field, "") for field in RAW_FIELDS})


def parse_metrics(text: str):
    result_lines = [
        line for line in text.splitlines()
        if "Loss" in line and "Macro F1" in line and "Weighted F1" in line
    ]
    if not result_lines:
        return {}

    def last_number(segment):
        values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", segment)
        return float(values[-1]) if values else None

    line = result_lines[-1]
    parts = [part.strip() for part in line.split("|")]
    if "Test Loss" in line and "Test Acc" in line and len(parts) >= 4:
        loss = last_number(parts[0])
        acc = last_number(parts[1])
        macro_f1 = last_number(parts[2])
        weighted_f1 = last_number(parts[3])
    elif len(parts) >= 4:
        acc = last_number(parts[0])
        loss = last_number(parts[1])
        macro_f1 = last_number(parts[2])
        weighted_f1 = last_number(parts[3])
    else:
        return {}

    if None in (acc, loss, macro_f1, weighted_f1):
        return {}
    if acc <= 1.0:
        acc *= 100.0
    return {
        "acc": acc,
        "loss": loss,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def log_tail(path: Path, max_lines=40):
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def guess_model_path(job: Job, out_dir: Path):
    if job.method in MCFD_ML_ABLATIONS:
        if job.method == "MEDG":
            name = f"task{job.task}_{job.seed}.pt"
        else:
            name = f"{job.method}_task{job.task}_{job.seed}.pt"
        return str(out_dir / "models" / name)
    patterns = {
        "MLDG": f"mldg_task{job.task}_{job.seed}.pt",
        "CDDG": f"cddg_task{job.task}_{job.seed}.pt",
        "M-DANN": f"mdann_task{job.task}_{job.seed}.pt",
    }
    name = patterns.get(job.method)
    return str(out_dir / "models" / name) if name else ""


def start_job(job: Job, gpu, out_dir: Path, estimated_gpu_memory_mb=0):
    run_dir = out_dir / "runs" / job.method / f"task{job.task}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "runtime_configs" / f"{job.method}_task{job.task}_seed{job.seed}.json"
    log_path = run_dir / f"seed{job.seed}.log"

    runtime_config = dict(job.runtime_config)
    runtime_config["_seed"] = job.seed
    runtime_config["_repeat"] = job.repeat
    runtime_config["_gpu"] = gpu
    runtime_config["OUTPUT_ROOT"] = str(out_dir)
    runtime_config["LOGS_DIR"] = str(out_dir / "logs")
    runtime_config["MODELS_DIR"] = str(out_dir / "models")
    runtime_config["FIGURES_DIR"] = str(out_dir / "figures")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2, ensure_ascii=False)

    env = os.environ.copy()
    env["MCED_RUNTIME_CONFIG"] = str(config_path)
    env["LOKY_MAX_CPU_COUNT"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    script = SRC_DIR / METHOD_SCRIPTS[job.method]
    cmd = [sys.executable, "-u", str(script), "--seed", str(job.seed)]
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    start_time = datetime.now()
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "job": job,
        "gpu": gpu,
        "estimated_gpu_memory_mb": estimated_gpu_memory_mb,
        "process": process,
        "log_file": log_file,
        "log_path": log_path,
        "start_time": start_time,
    }


def finish_job(active, raw_csv: Path, out_dir: Path):
    process = active["process"]
    process.wait()
    active["log_file"].close()
    end_time = datetime.now()
    job = active["job"]
    log_path = active["log_path"]
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = parse_metrics(text)
    status = "success" if process.returncode == 0 and metrics else "failed"
    duration = (end_time - active["start_time"]).total_seconds()
    row = {
        "method": job.method,
        "dataset": job.dataset,
        "task": job.task,
        "repeat": job.repeat,
        "seed": job.seed,
        "gpu": "" if active["gpu"] is None else active["gpu"],
        "status": status,
        "log_path": str(log_path),
        "model_path": guess_model_path(job, out_dir),
        "start_time": active["start_time"].isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "duration_sec": f"{duration:.1f}",
        "return_code": process.returncode,
    }
    row.update(metrics)
    append_raw(raw_csv, row)
    return row


def summarize(raw_csv: Path, summary_csv: Path):
    groups = {}
    if not raw_csv.exists():
        return
    with open(raw_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "success":
                continue
            key = (row["method"], row["dataset"], row["task"])
            groups.setdefault(key, []).append(row)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for (method, dataset, task), rows in sorted(groups.items()):
            out = {
                "method": method,
                "dataset": dataset,
                "task": task,
                "n_success": len(rows),
            }
            for metric in ["acc", "weighted_f1", "macro_f1", "loss"]:
                values = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
                out[f"{metric}_mean"] = f"{statistics.mean(values):.6f}" if values else ""
                out[f"{metric}_std"] = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "0.000000"
            writer.writerow(out)


def run(config, args):
    jobs = expand_jobs(config)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    validate_data_dirs(jobs)
    gpus = resolve_gpus(config)
    scheduler = scheduler_config(config)
    memory_mode = scheduler["mode"] == "memory" and any(gpu is not None for gpu in gpus)
    slots = fixed_slots(config, gpus) if not memory_mode else []

    if args.dry_run:
        if memory_mode:
            memory = nvidia_smi_memory(gpus)
            print(
                f"jobs={len(jobs)} gpus={gpus} scheduler=memory "
                f"cap_per_gpu={scheduler['max_jobs_per_gpu_cap']}"
            )
            if memory:
                print("gpu_memory=" + ", ".join(
                    f"gpu{gpu}:free={stats['free_mb']}MB,total={stats['total_mb']}MB"
                    for gpu, stats in sorted(memory.items())
                ))
            else:
                print("gpu_memory=unavailable; run will fall back to fixed scheduling")
        else:
            print(
                f"jobs={len(jobs)} gpus={gpus} scheduler=fixed "
                f"max_jobs_per_gpu={config.get('max_jobs_per_gpu', 1)} slots={len(slots)}"
            )
        dataset_counts = Counter((job.dataset, job.channels) for job in jobs)
        task_counts = Counter((job.task, job.dataset, job.channels) for job in jobs)
        defaults = config.get("defaults", {})
        print(
            "dataloader="
            f"num_workers={defaults.get('num_workers', 0)}, "
            f"pin_memory={defaults.get('pin_memory', True)}, "
            f"persistent_workers={defaults.get('persistent_workers', True)}, "
            f"prefetch_factor={defaults.get('prefetch_factor', 2)}"
        )
        print("datasets=" + ", ".join(
            f"{dataset}/channels={channels}: {count}" for (dataset, channels), count in sorted(dataset_counts.items())
        ))
        print("tasks=" + ", ".join(
            f"task{task}:{dataset}/channels={channels}" for (task, dataset, channels), _ in sorted(task_counts.items())
        ))
        for job in jobs[:20]:
            print(
                f"{job.method} task={job.task} dataset={job.dataset} "
                f"channels={job.channels} repeat={job.repeat} seed={job.seed} "
                f"data_dir={job.data_dir} estimated_gpu_memory_mb={job_memory_mb(config, job)}"
            )
        if len(jobs) > 20:
            print(f"... {len(jobs) - 20} more jobs")
        return

    out_dir = make_output_dir(config)
    raw_csv = out_dir / "raw_runs.csv"
    summary_csv = out_dir / "summary.csv"
    ensure_raw_csv(raw_csv)
    completed = load_completed(raw_csv) if args.resume else set()
    queue = [job for job in jobs if (job.method, job.task, job.seed) not in completed]
    running = []

    print(f"Output: {out_dir}")
    if memory_mode and not nvidia_smi_memory(gpus):
        memory_mode = False
        slots = fixed_slots(config, gpus)
        print("GPU memory query unavailable; falling back to fixed scheduling.")
    if memory_mode:
        print(
            f"Queued jobs: {len(queue)} | GPUs: {gpus} | scheduler=memory | "
            f"cap_per_gpu={scheduler['max_jobs_per_gpu_cap']}"
        )
    else:
        print(f"Queued jobs: {len(queue)} | GPUs: {gpus} | scheduler=fixed | slots={len(slots)}")

    while queue or running:
        if memory_mode:
            launched = True
            while queue and launched:
                launched = False
                memory = nvidia_smi_memory(gpus)
                if not memory and not running:
                    memory_mode = False
                    slots = fixed_slots(config, gpus)
                    print("GPU memory query unavailable; switching to fixed scheduling.")
                    break
                slot_gpu = choose_memory_gpu(config, queue[0], gpus, running, memory)
                if slot_gpu is not None:
                    job = queue.pop(0)
                    estimate = job_memory_mb(config, job)
                    active = start_job(job, slot_gpu, out_dir, estimate)
                    running.append(active)
                    launched = True
                    print(
                        f"START {job.method} task={job.task} seed={job.seed} gpu={slot_gpu} "
                        f"estimated_gpu_memory_mb={estimate}"
                    )
                elif not running and queue and memory:
                    slot_gpu = max(memory.items(), key=lambda item: item[1]["free_mb"])[0]
                    job = queue.pop(0)
                    estimate = job_memory_mb(config, job)
                    active = start_job(job, slot_gpu, out_dir, estimate)
                    running.append(active)
                    launched = True
                    print(
                        f"START {job.method} task={job.task} seed={job.seed} gpu={slot_gpu} "
                        f"estimated_gpu_memory_mb={estimate} forced_single_job=true"
                    )
        else:
            while queue and len(running) < len(slots):
                slot_gpu = slots[len(running) % len(slots)]
                job = queue.pop(0)
                estimate = job_memory_mb(config, job)
                active = start_job(job, slot_gpu, out_dir, estimate)
                running.append(active)
                print(f"START {job.method} task={job.task} seed={job.seed} gpu={slot_gpu}")

        time.sleep(float(scheduler["poll_interval_sec"]))
        still_running = []
        for active in running:
            if active["process"].poll() is None:
                still_running.append(active)
                continue
            row = finish_job(active, raw_csv, out_dir)
            print(
                f"DONE {row['method']} task={row['task']} seed={row['seed']} "
                f"status={row['status']} return_code={row.get('return_code', '')} "
                f"acc={row.get('acc', '')} wf1={row.get('weighted_f1', '')}"
            )
            if row["status"] == "failed":
                tail = log_tail(Path(row["log_path"]))
                if tail:
                    print(f"FAILED LOG TAIL {row['log_path']}\n{tail}\nEND FAILED LOG TAIL")
            summarize(raw_csv, summary_csv)
        running = still_running

    summarize(raw_csv, summary_csv)
    print(f"Raw results: {raw_csv}")
    print(f"Summary: {summary_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments" / "auto_train.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    run(config, args)


if __name__ == "__main__":
    main()
