#!/usr/bin/env bash
set -euo pipefail

# Run warmup sweeps for the Adagrad optimizer on Tiny Shakespeare.
# Tests warmup_iters values 0, 100, 250, 500 with a fixed learning rate (1e-2)
# and total training steps (max_iters) of 5000. Collects training losses,
# validation losses, and gradient norms from metrics.jsonl into a CSV summary.

CONFIG=${1:-config/train_shakespeare_char.py}
RESULTS_ROOT=${RESULTS_ROOT:-out/adagrad_warmup_sweep}
RESULTS_FILE=${RESULTS_FILE:-"${RESULTS_ROOT}/adagrad_warmup_results.csv"}

WARMUPS=(0 100 250 500)
LEARNING_RATE=1e-2
MAX_ITERS=5000
EXTRA_ARGS=${EXTRA_ARGS:-"--decay_lr=True"}

mkdir -p "${RESULTS_ROOT}"
printf "optimizer,warmup_iters,out_dir,best_val_loss,train_losses,val_losses,grad_norms\n" > "${RESULTS_FILE}"

for warmup in "${WARMUPS[@]}"; do
  run_dir="${RESULTS_ROOT}/warmup_${warmup}"
  mkdir -p "${run_dir}"

  echo "[adagrad] warmup_iters=${warmup} -> ${run_dir}"
  python train.py "${CONFIG}" \
    --optimizer_name="adagrad" \
    --learning_rate="${LEARNING_RATE}" \
    --warmup_iters="${warmup}" \
    --max_iters="${MAX_ITERS}" \
    --lr_decay_iters="${MAX_ITERS}" \
    --out_dir="${run_dir}" \
    --out_dir_by_optimizer=False \
    ${EXTRA_ARGS}

  python - "$run_dir" "$RESULTS_FILE" "$warmup" <<'PY'
import csv
import json
import os
import sys
import torch

run_dir, results_file, warmup = sys.argv[1:4]
ckpt_path = os.path.join(run_dir, "ckpt.pt")
metrics_path = os.path.join(run_dir, "metrics.jsonl")

train_losses = []
val_losses = []
grad_norms = []

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            if "train_loss" in rec:
                train_losses.append(float(rec["train_loss"]))
            if "val_loss" in rec:
                val_losses.append(float(rec["val_loss"]))
            if "grad_norm" in rec:
                grad_norms.append(float(rec["grad_norm"]))
else:
    print(f"[WARN] No metrics file at {metrics_path}; logging empty series.")

if not os.path.exists(ckpt_path):
    val_loss = float("nan")
    print(f"[WARN] No checkpoint found at {ckpt_path}; writing NaN.")
else:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    val_loss = float(ckpt.get("best_val_loss", float("nan")))
    print(f"[adagrad] warmup={warmup} best_val_loss={val_loss}")

with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "adagrad",
        warmup,
        run_dir,
        val_loss,
        json.dumps(train_losses),
        json.dumps(val_losses),
        json.dumps(grad_norms),
    ])

summary_path = os.path.join(run_dir, "warmup_metrics_summary.json")
with open(summary_path, "w") as f:
    json.dump(
        {
            "optimizer": "adagrad",
            "warmup_iters": int(warmup),
            "out_dir": run_dir,
            "best_val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "grad_norms": grad_norms,
        },
        f,
        indent=2,
    )
print(f"saved JSON metrics to {summary_path}")
PY
done

echo "Done. Results saved to ${RESULTS_FILE}."