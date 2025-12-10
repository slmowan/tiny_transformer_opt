#!/usr/bin/env bash
set -euo pipefail

# Sweep learning rates for multiple optimizers on a given config (default: Tiny Shakespeare).
# This script runs train.py repeatedly, reads metrics from each checkpoint and metrics log,
# and writes a CSV with columns:
# optimizer,learning_rate,out_dir,best_val_loss,train_losses,val_losses,times_ms,mfu_percent.

CONFIG=${1:-config/train_shakespeare_char.py}
RESULTS_ROOT=${RESULTS_ROOT:-out/lr_sweep}
RESULTS_FILE=${RESULTS_FILE:-"${RESULTS_ROOT}/lr_sensitivity_results.csv"}

# Accept OPTIMIZERS as a whitespace- or comma-separated list and split into an array.
OPTIMIZERS_RAW=${OPTIMIZERS:-"adamw adam momentum sgd adagrad"}
OPTIMIZERS_RAW=${OPTIMIZERS_RAW//,/ }
read -r -a OPTIMIZERS <<< "${OPTIMIZERS_RAW}"

EXTRA_ARGS=${EXTRA_ARGS:-"--decay_lr=False --grad_clip=1.0"}

mkdir -p "${RESULTS_ROOT}"
# overwrite the results file on each run to avoid mixing experiments
printf "optimizer,learning_rate,out_dir,best_val_loss,train_losses,val_losses,times_ms,mfu_percent\n" > "${RESULTS_FILE}"

for opt in "${OPTIMIZERS[@]}"; do
  case "${opt}" in
    adamw)
      lr_grid=(6e-4 3e-4 1e-3 2e-3 4e-4)
      ;;
    adam)
      lr_grid=(1e-3 5e-4 2e-3 3e-4 1e-4)
      ;;
    momentum)
      lr_grid=(1e-2 5e-3 2e-2 5e-4 1e-3)
      ;;
    sgd)
      lr_grid=(1e-2 5e-3 2e-2 1e-3 5e-4)
      ;;
    adagrad)
      lr_grid=(1e-2 5e-3 2e-2 1e-3 5e-4)
      ;;
    *)
      echo "Unsupported optimizer: ${opt}" >&2
      exit 1
      ;;
  esac

  for lr in "${lr_grid[@]}"; do
    run_dir="${RESULTS_ROOT}/${opt}/lr_${lr}"
    mkdir -p "${run_dir}"

    echo "[${opt}] lr=${lr} -> ${run_dir}"
    python train.py "${CONFIG}" \
      --optimizer_name="${opt}" \
      --learning_rate="${lr}" \
      --out_dir="${run_dir}" \
      --out_dir_by_optimizer=False \
      ${EXTRA_ARGS}

    python - "$run_dir" "$RESULTS_FILE" "$opt" "$lr" <<'PY'
import csv
import json
import os
import sys
import torch

run_dir, results_file, opt, lr = sys.argv[1:5]
ckpt_path = os.path.join(run_dir, "ckpt.pt")
metrics_path = os.path.join(run_dir, "metrics.jsonl")

train_losses = []
val_losses = []
times_ms = []
mfu_percent = []

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            if "train_loss" in rec:
                train_losses.append(float(rec["train_loss"]))
            if "val_loss" in rec:
                val_losses.append(float(rec["val_loss"]))
            if "time_ms" in rec:
                times_ms.append(float(rec["time_ms"]))
            if "mfu" in rec:
                mfu_percent.append(float(rec["mfu"]))
else:
    print(f"[WARN] No metrics file at {metrics_path}; logging empty series.")

if not os.path.exists(ckpt_path):
    val_loss = float("nan")
    print(f"[WARN] No checkpoint found at {ckpt_path}; writing NaN.")
else:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    val_loss = float(ckpt.get("best_val_loss", float("nan")))
    print(f"[{opt}] lr={lr} best_val_loss={val_loss}")

with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)
    train_json = json.dumps(train_losses)
    val_json = json.dumps(val_losses)
    times_json = json.dumps(times_ms)
    mfu_json = json.dumps(mfu_percent)
    writer.writerow([
        opt,
        lr,
        run_dir,
        val_loss,
        train_json,
        val_json,
        times_json,
        mfu_json,
    ])

summary_path = os.path.join(run_dir, "lr_metrics_summary.json")
with open(summary_path, "w") as f:
    json.dump(
        {
            "optimizer": opt,
            "learning_rate": lr,
            "out_dir": run_dir,
            "best_val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "times_ms": times_ms,
            "mfu_percent": mfu_percent,
        },
        f,
        indent=2,
    )
print(f"saved JSON metrics to {summary_path}")
PY
  done
done

echo "Done. Results saved to ${RESULTS_FILE}".
