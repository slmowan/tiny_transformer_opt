# tiny_transformer_opt

A minimal GPT-style transformer implementation and training harness optimized for efficient experimentation. The code is self-contained and follows the structure popularized by nanoGPT, exposing configuration flags for model shape, optimizer, and distributed training.

## Project layout
- `model.py`: GPT model definition including attention, MLP blocks, optimizer setup, and text generation utilities.
- `train.py`: Training entrypoint that supports single-GPU debugging or distributed data parallel runs via `torchrun` with configurable hyperparameters.
- `config/`: Example configuration overrides for GPT-2 scale models, Shakespeare finetuning, and evaluation scripts.
- `sample.py` / `bench.py`: Sampling and benchmarking helpers for quick experimentation.

## Quickstart
1. **Install dependencies** (PyTorch 2.x recommended for flash attention support):
   ```bash
   pip install torch transformers tiktoken
   ```
2. **Train on a single GPU** with default GPT-2 124M-style settings:
   ```bash
   python train.py --batch_size=32 --compile=False
   ```
3. **Distributed training** example on 4 GPUs (one node):
   ```bash
   torchrun --standalone --nproc_per_node=4 train.py
   ```
4. **Load pretrained GPT-2 weights** for evaluation or finetuning by setting `init_from`:
   ```bash
   python train.py --init_from=gpt2 --eval_only=True
   ```

## Key features
- Flash attention is used when available (`torch.nn.functional.scaled_dot_product_attention`), falling back to a masked implementation on older PyTorch versions.
- Weight tying between token embeddings and output head for parameter efficiency.
- Optimizer groups automatically split decayed/non-decayed parameters and enable fused AdamW when running on CUDA; switchable optimizers include AdamW, Adam, SGD, Momentum SGD, and Adagrad via the `optimizer_name` flag.
- Generation helper (`GPT.generate`) supports temperature scaling and top-k filtering for sampling text.

## Configuration hints
Hyperparameters in `train.py` can be overridden via command-line flags or config files executed through `configurator.py`. Example configs under `config/` include GPT-2 size presets (`train_gpt2.py`, `eval_gpt2*.py`) and Shakespeare character-level finetuning (`train_shakespeare_char.py`).

## Run Tiny Shakespeare with different optimizers
Use the built-in Shakespeare character-level config and override the optimizer per run. All commands read from `config/train_shakespeare_char.py`, which sets the dataset path and model size; only the optimizer changes between runs.

```bash
# AdamW (default)
python train.py config/train_shakespeare_char.py --optimizer_name=adamw

# Adam (no weight decay decoupling)
python train.py config/train_shakespeare_char.py --optimizer_name=adam

# SGD with constant step size
python train.py config/train_shakespeare_char.py --optimizer_name=sgd --learning_rate=1e-2

# Momentum SGD (tunable momentum)
python train.py config/train_shakespeare_char.py --optimizer_name=momentum --momentum=0.9 --learning_rate=1e-2

# Adagrad
python train.py config/train_shakespeare_char.py --optimizer_name=adagrad --learning_rate=1e-2
```

Tips:
- Checkpoints are automatically written to optimizer-specific folders (e.g., `out/adamw/ckpt.pt`, `out/sgd/ckpt.pt`). Disable this behavior with `--out_dir_by_optimizer=False` if you want everything in a single directory.
- Add `--grad_clip=1.0` (enabled by default) or adjust the value to compare stability with/without clipping.
- Append `--decay_lr=False` to keep a fixed learning rate if you want a pure optimizer comparison without scheduling.

## Learning rate sensitivity sweep (per optimizer)
Use the helper script to sweep a reasonable set of learning rates for each optimizer, write checkpoints to separate folders, and log best validation losses to a CSV for plotting `val_loss` vs. `learning_rate`.

```bash
# Default: Tiny Shakespeare config, optimizer-specific learning rate grids
./scripts/lr_sweep.sh

# Optional: override config or pass extra args (e.g., more iterations)
RESULTS_ROOT=out/lr_sweep_long \
EXTRA_ARGS="--decay_lr=False --max_iters=7000" \
./scripts/lr_sweep.sh config/train_shakespeare_char.py
```

Outputs:
- Per-run checkpoints under `out/lr_sweep/<optimizer>/lr_<value>/ckpt.pt` (no automatic optimizer subfolder nesting is added by the script).
- Aggregated CSV at `out/lr_sweep/lr_sensitivity_results.csv` with columns: `optimizer,learning_rate,out_dir,best_val_loss,train_losses,val_losses,times_ms,mfu_percent`.
  - `train_losses` and `val_losses` are JSON-encoded arrays logged at each evaluation step.
  - `times_ms` and `mfu_percent` capture per-logging-interval step time (ms) and Model FLOPs Utilization (%) so you can study throughput vs. learning rate.
  - Raw per-step metrics are also stored as JSONL at `out/lr_sweep/<optimizer>/lr_<value>/metrics.jsonl` for custom analysis.

You can modify `OPTIMIZERS` or the hardcoded learning rate grids inside `scripts/lr_sweep.sh` to explore additional candidates.
`OPTIMIZERS` accepts a space- or comma-separated list (e.g., `OPTIMIZERS="adamw,adam,sgd" ./scripts/lr_sweep.sh`).

