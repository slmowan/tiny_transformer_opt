#!/usr/bin/env bash
set -euo pipefail


cd "$(dirname "$0")/.."
echo "[INFO] Working directory: $(pwd)"


CONFIG=${1:-config/train_shakespeare_char.py}

RESULTS_ROOT=${RESULTS_ROOT:-out/weight_decay}
RESULTS_FILE=${RESULTS_FILE:-"${RESULTS_ROOT}/weight_decay_results.csv"}

OPTIMIZERS_RAW=${OPTIMIZERS:-"adam adamw"}
OPTIMIZERS_RAW=${OPTIMIZERS_RAW//,/ }
read -r -a OPTIMIZERS <<< "${OPTIMIZERS_RAW}"

WEIGHT_DECAYS_RAW=${WEIGHT_DECAYS:-"0.0 1e-4 1e-3 1e-2 5e-2 1e-1"}
WEIGHT_DECAYS_RAW=${WEIGHT_DECAYS_RAW//,/ }
read -r -a WEIGHT_DECAYS <<< "${WEIGHT_DECAYS_RAW}"

EXTRA_ARGS=${EXTRA_ARGS:-"--decay_lr=False --grad_clip=1.0"}

mkdir -p "${RESULTS_ROOT}"
printf "optimizer,weight_decay,out_dir,best_val_loss,train_losses,val_losses,times_ms,mfu_percent\n" > "${RESULTS_FILE}"

echo "[INFO] Results root: ${RESULTS_ROOT}"
echo "[INFO] Config file:  ${CONFIG}"
echo "[INFO] Optimizers:   ${OPTIMIZERS[@]}"
echo "[INFO] Weight decay: ${WEIGHT_DECAYS[@]}"
echo ""

for opt in "${OPTIMIZERS[@]}"; do
  if [[ "${opt}" != "adam" && "${opt}" != "adamw" ]]; then
    echo "[WARN] Skipping unsupported optimizer: ${opt}"
    continue
  fi

  for wd in "${WEIGHT_DECAYS[@]}"; do
    run_dir="${RESULTS_ROOT}/${opt}/wd_${wd}"
    mkdir -p "${run_dir}"

    echo "=========================================================="
    echo "[RUN] optimizer=${opt}, weight_decay=${wd}"
    echo "[DIR] ${run_dir}"
    echo "=========================================================="

    LOG_FILE="${run_dir}/train.log"

    echo "[INFO] Training started..." | tee "${LOG_FILE}"

    if python train.py "${CONFIG}" \
      --optimizer_name="${opt}" \
      --weight_decay="${wd}" \
      --out_dir="${run_dir}" \
      --out_dir_by_optimizer=False \
      ${EXTRA_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    then
      echo "[INFO] Training finished." | tee -a "${LOG_FILE}"
    else
      echo "[ERROR] Training crashed! See log: ${LOG_FILE}"
    fi

    python - "${run_dir}" "${RESULTS_FILE}" "${opt}" "${wd}" <<'PY'
import csv
import json
import os
import sys
import torch

run_dir, results_file, opt, wd = sys.argv[1:5]

ckpt_path = os.path.join(run_dir, "ckpt.pt")
metrics_path = os.path.join(run_dir, "metrics.jsonl")

train_losses, val_losses, times_ms, mfu_percent = [], [], [], []

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        for line in f:
            rec = json.loads(line)
            train_losses.append(float(rec.get("train_loss", float("nan"))))
            val_losses.append(float(rec.get("val_loss", float("nan"))))
            times_ms.append(float(rec.get("time_ms", float("nan"))))
            mfu_percent.append(float(rec.get("mfu", float("nan"))))
else:
    print(f"[WARN] No metrics.jsonl found at: {metrics_path}")

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    val_loss = float(ckpt.get("best_val_loss", float("nan")))
else:
    print(f"[WARN] No checkpoint file found: {ckpt_path}")
    val_loss = float("nan")

with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        opt,
        wd,
        run_dir,
        val_loss,
        json.dumps(train_losses),
        json.dumps(val_losses),
        json.dumps(times_ms),
        json.dumps(mfu_percent),
    ])

summary_path = os.path.join(run_dir, "weight_decay_metrics_summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "optimizer": opt,
        "weight_decay": wd,
        "out_dir": run_dir,
        "best_val_loss": val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "times_ms": times_ms,
        "mfu_percent": mfu_percent,
    }, f, indent=2)

print(f"[INFO] Summary saved to {summary_path}")
PY

  done
done

echo ""
echo "=========================================================="
echo "[DONE] Sweep completed. CSV summary saved to:"
echo "       ${RESULTS_FILE}"
echo "=========================================================="
