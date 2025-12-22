from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiments in sequence")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest or specified run-id")
    parser.add_argument("--resume-dir", type=str, default=None, help="Resume from an explicit run directory")
    parser.add_argument("--precompute-workers", type=int, default=1)
    parser.add_argument("--skip-dsp-compare", action="store_true")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-retrieval-ml", action="store_true")
    parser.add_argument("--skip-panns", action="store_true")
    parser.add_argument("--skip-ast", action="store_true")
    parser.add_argument("--skip-clap", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--gemini-sleep", type=float, default=0.0)
    parser.add_argument("--gemini-max-samples", type=int, default=None)
    parser.add_argument("--gemini-workers", type=int, default=1)
    parser.add_argument(
        "--gemini-prompt-style",
        type=str,
        default="simple",
        choices=["simple", "guided"],
        help="Prompt style for Gemini zero-shot.",
    )
    parser.add_argument("--clap-batch-size", type=int, default=8)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as f:
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, env=env)
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        return process.wait()


def find_latest_run(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    candidates = [p for p in results_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    results_root = Path(args.results_dir)
    base_dir: Path
    run_id: str
    if args.resume_dir:
        base_dir = Path(args.resume_dir)
        run_id = base_dir.name
    elif args.resume:
        if args.run_id:
            base_dir = results_root / args.run_id
            run_id = args.run_id
        else:
            latest = find_latest_run(results_root)
            if latest is None:
                raise FileNotFoundError(f"No existing runs under {results_root}")
            base_dir = latest
            run_id = latest.name
    else:
        run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        base_dir = results_root / run_id
    logs_dir = base_dir / "logs"
    features_dir = base_dir / "features"
    history_dir = base_dir / "history"
    models_dir = base_dir / "models"
    retrieval_dir = base_dir / "retrieval"
    validation_dir = base_dir / "validation"
    predictions_dir = base_dir / "predictions"

    base_dir.mkdir(parents=True, exist_ok=True)

    retrieval_cfg = cfg.get("retrieval", {})
    frame_lengths = retrieval_cfg.get("frame_lengths", [512, 1024, 2048])
    hop_lengths = retrieval_cfg.get("hop_lengths", [256, 512, 1024])
    n_mels = retrieval_cfg.get("n_mels", 40)
    n_mfcc = retrieval_cfg.get("n_mfcc", 13)

    classification_cfg = cfg.get("classification", {})
    clf_frame_length = classification_cfg.get("frame_length", 1024)
    clf_hop_length = classification_cfg.get("hop_length", 512)
    clf_n_mels = classification_cfg.get("n_mels", 40)
    clf_epochs = classification_cfg.get("epochs", 30)
    clf_batch_size = classification_cfg.get("batch_size", 32)
    clf_lr = classification_cfg.get("lr", 1e-3)

    steps = []

    if not args.skip_dsp_compare:
        steps.append(
            {
                "name": "dsp_compare",
                "cmd": [
                    "python",
                    "scripts/tools/compare_librosa.py",
                    "--output",
                    str(validation_dir / "librosa_compare.json"),
                ],
            }
        )

    if not args.skip_precompute:
        for fl in frame_lengths:
            for hl in hop_lengths:
                steps.append(
                    {
                        "name": f"precompute_mfcc_fl{fl}_hl{hl}",
                        "cmd": [
                            "python",
                            "scripts/tools/precompute_features.py",
                            "--feature-types",
                            "mfcc",
                            "--frame-length",
                            str(fl),
                            "--hop-length",
                            str(hl),
                            "--n-mels",
                            str(n_mels),
                            "--n-mfcc",
                            str(n_mfcc),
                            "--feature-cache",
                            str(features_dir),
                            "--workers",
                            str(args.precompute_workers),
                        ],
                    }
                )
        steps.append(
            {
                "name": "precompute_log_mel",
                "cmd": [
                    "python",
                    "scripts/tools/precompute_features.py",
                    "--feature-types",
                    "log_mel",
                    "--frame-length",
                    str(clf_frame_length),
                    "--hop-length",
                    str(clf_hop_length),
                    "--n-mels",
                    str(clf_n_mels),
                    "--feature-cache",
                    str(features_dir),
                    "--workers",
                    str(args.precompute_workers),
                ],
            }
        )

    if not args.skip_retrieval:
        steps.append(
            {
                "name": "retrieval_mfcc",
                "cmd": [
                    "python",
                    "scripts/tasks/run_retrieval.py",
                    "--frame-lengths",
                    *[str(v) for v in frame_lengths],
                    "--hop-lengths",
                    *[str(v) for v in hop_lengths],
                    "--n-mels",
                    str(n_mels),
                    "--n-mfcc",
                    str(n_mfcc),
                    "--feature-cache",
                    str(features_dir),
                    "--output",
                    str(retrieval_dir / "retrieval_mfcc.csv"),
                ],
            }
        )

    if not args.skip_cnn:
        steps.append(
            {
                "name": "train_cnn",
                "cmd": [
                    "python",
                    "scripts/models/train_cnn.py",
                    "--frame-length",
                    str(clf_frame_length),
                    "--hop-length",
                    str(clf_hop_length),
                    "--n-mels",
                    str(clf_n_mels),
                    "--epochs",
                    str(clf_epochs),
                    "--batch-size",
                    str(clf_batch_size),
                    "--lr",
                    str(clf_lr),
                    "--output",
                    str(models_dir / "cnn.pt"),
                    "--history",
                    str(history_dir / "train_cnn.csv"),
                    "--feature-cache",
                    str(features_dir),
                ],
            }
        )

    if not args.skip_retrieval_ml:
        steps.append(
            {
                "name": "retrieval_ml_cnn",
                "cmd": [
                    "python",
                    "scripts/tasks/run_retrieval_ml.py",
                    "--model-type",
                    "cnn",
                    "--model",
                    str(models_dir / "cnn.pt"),
                    "--batch-size",
                    str(args.retrieval_batch_size),
                    "--feature-cache",
                    str(features_dir),
                ],
            }
        )

    if not args.skip_panns:
        steps.append(
            {
                "name": "panns_transfer",
                "cmd": [
                    "python",
                    "scripts/models/eval_panns_transfer.py",
                    "--history",
                    str(history_dir / "panns_transfer.csv"),
                    "--output",
                    str(models_dir / "panns_transfer.pt"),
                ],
            }
        )
        if not args.skip_retrieval_ml:
            steps.append(
                {
                    "name": "retrieval_ml_panns",
                    "cmd": [
                        "python",
                        "scripts/tasks/run_retrieval_ml.py",
                        "--model-type",
                        "panns",
                        "--batch-size",
                        str(args.retrieval_batch_size),
                        "--feature-cache",
                        str(features_dir),
                    ],
                }
            )

    if not args.skip_ast:
        steps.append(
            {
                "name": "ast_transfer",
                "cmd": [
                    "python",
                    "scripts/models/eval_ast_transfer.py",
                    "--history",
                    str(history_dir / "ast_transfer.csv"),
                    "--output",
                    str(models_dir / "ast_transfer.pt"),
                ],
            }
        )
        if not args.skip_retrieval_ml:
            steps.append(
                {
                    "name": "retrieval_ml_ast",
                    "cmd": [
                        "python",
                        "scripts/tasks/run_retrieval_ml.py",
                        "--model-type",
                        "ast",
                        "--model-id",
                        "MIT/ast-finetuned-audioset-10-10-0.4593",
                        "--batch-size",
                        str(args.retrieval_batch_size),
                        "--feature-cache",
                        str(features_dir),
                    ],
                }
            )

    if not args.skip_clap:
        steps.append(
            {
                "name": "clap_transfer",
                "cmd": [
                    "python",
                    "scripts/models/eval_clap_transfer.py",
                    "--history",
                    str(history_dir / "clap_transfer.csv"),
                    "--feature-cache",
                    str(features_dir),
                    "--output",
                    str(models_dir / "clap_transfer.pt"),
                ],
            }
        )
        steps.append(
            {
                "name": "clap_zeroshot",
                "cmd": [
                    "python",
                    "scripts/models/eval_clap_zeroshot.py",
                    "--model",
                    "laion/clap-htsat-unfused",
                    "--batch-size",
                    str(args.clap_batch_size),
                    "--output",
                    str(predictions_dir / "clap_zeroshot.csv"),
                ],
            }
        )
        if not args.skip_retrieval_ml:
            steps.append(
                {
                    "name": "retrieval_ml_clap",
                    "cmd": [
                        "python",
                        "scripts/tasks/run_retrieval_ml.py",
                        "--model-type",
                        "clap",
                        "--model-id",
                        "laion/clap-htsat-unfused",
                        "--batch-size",
                        str(args.retrieval_batch_size),
                        "--feature-cache",
                        str(features_dir),
                    ],
                }
            )

    if not args.skip_gemini:
        gemini_style = args.gemini_prompt_style
        gemini_suffix = "" if gemini_style == "simple" else f"_{gemini_style}"
        gemini_predictions = predictions_dir / f"llm_predictions{gemini_suffix}.csv"
        gemini_cmd = [
            "python",
            "scripts/models/eval_gemini_zeroshot.py",
            "--model",
            "gemini-3-flash-preview",
            "--sleep",
            str(args.gemini_sleep),
            "--workers",
            str(args.gemini_workers),
            "--output",
            str(gemini_predictions),
            "--resume",
            "--prompt-style",
            gemini_style,
        ]
        if args.gemini_max_samples:
            gemini_cmd.extend(["--max-samples", str(args.gemini_max_samples)])
        steps.append({"name": f"gemini_zeroshot_{gemini_style}", "cmd": gemini_cmd})
        steps.append(
            {
                "name": f"gemini_eval_{gemini_style}",
                "cmd": [
                    "python",
                    "scripts/tasks/eval_llm_baseline.py",
                    "--predictions",
                    str(gemini_predictions),
                ],
            }
        )

    if not args.skip_plot:
        steps.append(
            {
                "name": "plot_history",
                "cmd": [
                    "python",
                    "scripts/tools/plot_history.py",
                    "--history",
                    str(history_dir / "train_cnn.csv"),
                    "--output",
                    str(base_dir / "plots" / "cnn_history.png"),
                ],
            }
        )

    summary_path = base_dir / "summary.json"
    summary = {"run_id": run_id, "results_dir": str(base_dir), "steps": []}
    completed = {}
    if args.resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        for step in summary.get("steps", []):
            if step.get("return_code") == 0:
                completed[step.get("name")] = step
        if args.gemini_prompt_style == "simple":
            if "gemini_zeroshot" in completed and "gemini_zeroshot_simple" not in completed:
                completed["gemini_zeroshot_simple"] = completed["gemini_zeroshot"]
            if "gemini_eval" in completed and "gemini_eval_simple" not in completed:
                completed["gemini_eval_simple"] = completed["gemini_eval"]
    updated_steps = []
    for step in steps:
        name = step["name"]
        if args.resume and name in completed:
            print(f"=== Skipping (already complete): {name} ===")
            updated_steps.append(completed[name])
            continue

        log_path = logs_dir / f"{name}.log"
        print(f"=== Running: {name} ===")
        code = run_cmd(step["cmd"], log_path)
        entry = {
            "name": name,
            "command": step["cmd"],
            "log": str(log_path),
            "return_code": code,
        }
        updated_steps.append(entry)
        if code != 0 and not args.continue_on_error:
            print(f"Step failed: {name} (exit {code})")
            break

    summary["run_id"] = run_id
    summary["results_dir"] = str(base_dir)
    summary["steps"] = updated_steps
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()
