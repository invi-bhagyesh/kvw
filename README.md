# KVW Stage A — Runbook

Fork of [KVW](https://arxiv.org/abs/2601.21794) with a per-sample redundancy
measurement added on top. Goal: test whether residual forget-specific firing
after KVW weakening predicts which samples the unlearned model still answers
correctly. See `docs/idea.md` for the motivating LinkedIn exchange and
`docs/progress_log.md` for current status.

Paper README contents: see the upstream repo at
[kyj93790/KVW](https://github.com/kyj93790/KVW).

---

## What this fork adds

- `CLEAR/baselines/KVW.py` — `compute_knowledge_coeffs_per_sample`,
  `residual_redundancy_per_sample`, and `--phase stage_a_measure`.
- `CLEAR/eval.py` — per-sample forget correctness logging.
- `CLEAR/stage_a.sh` — measure → eval → analysis pipeline.
- `CLEAR/finetune.py` — small bug fix in the `--gradient_accumulation` branch
  (upstream hardcoded a 4-tuple unpack on a dict batch; never exercised by
  shipped scripts).
- `scripts/stage_a_analysis.py` — AUC + quintile bins + plots, joined by
  identity name (not index).

---

## Pod setup

### GPU

**A100 40GB** on RunPod. Sweet spot. H100 works too (faster, more expensive);
L40S 48GB fine; RTX 4090 24GB only if you drop finetune to bs=1 with no
accumulation.

### Volume

100 GB network volume mounted at `/workspace`. Persists across restarts.
CLEAR dataset + two checkpoints + conda env ≈ 50–60 GB, so 100 GB leaves
headroom.

### Env

```bash
cd /workspace
git clone https://github.com/invi-bhagyesh/kvw.git KVW
cd KVW/CLEAR

# System deps (if template lacks them)
apt-get update -y && apt-get install -y \
    git-lfs tmux build-essential ninja-build libopenmpi-dev openmpi-bin
git lfs install

# Python env — prefer conda if available on the image
conda create -n clear python=3.10 -y && conda activate clear
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

**Verify you're in the right env** before running anything:

```bash
which python3
python3 -c "import transformers, peft; print(transformers.__version__, peft.__version__)"
# want: 4.57.1  0.17.1
```

If the RunPod image has baked-in system-level `transformers` / `peft` that
shadow the conda env, install into a venv explicitly instead of relying on
conda:

```bash
python3 -m venv /workspace/venv && source /workspace/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

---

## Data

The code hardcodes `data/CLEAR/...` paths. HF clones put data at
`KVW/CLEAR/CLEAR/` by default, so symlink it:

```bash
cd /workspace/KVW/CLEAR
mkdir -p data
git clone https://huggingface.co/datasets/therem/CLEAR data/CLEAR

# Optional but recommended — adds the _perturbed classification splits and
# the two-fold-val material the paper uses. Merge into the main folder:
cd data
git clone https://huggingface.co/datasets/yejinkim/clear-two-fold-val
cd clear-two-fold-val && mv * ../CLEAR/ && cd .. && rm -rf clear-two-fold-val
cd ..
```

Sanity check — should include `forget05`, `retain95`, `forget05+tofu`,
`retain95+tofu`, `full+tofu`, `forget05_perturbed`, `retain_perturbed`,
`real_faces`, `real_world`:

```bash
ls data/CLEAR/
```

---

## Running the pipeline

Run everything inside `tmux` so long jobs survive disconnects:

```bash
tmux new -s kvw
cd /workspace/KVW/CLEAR
conda activate clear   # or: source /workspace/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 1. Finetune vanilla model (~1.5–2 h)

Installs CLEAR's fictional-identity knowledge into Qwen2-VL-2B via LoRA.
This is the model KVW will later unlearn from. Skip the oracle run (not
needed for Stage A).

Do **not** pass `--gradient_accumulation` — its branch in upstream
`finetune.py` interacts badly with PEFT's frozen base + grad checkpointing,
and bs=1 fits in 40 GB without it.

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model_id Qwen/Qwen2-VL-2B-Instruct \
    --save_dir checkpoints/qwen2B_vanilla \
    --batch_size 1 \
    --lr 1e-4 \
    --num_epochs 1 \
    --forget_ratio 5 \
    --is_oracle False 2>&1 | tee logs_finetune.txt
```

### 2. Precompute KC_r (~15–20 min)

Forward pass over the retain set; records average |activation| per neuron.

```bash
bash compute_kc_r.sh 2>&1 | tee logs_kc_r.txt
```

Output: `kc/kc_r_retain_95.pt`.

### 3. Run KVW weakening (~30–60 min)

```bash
bash kvw.sh 2>&1 | tee logs_kvw.txt
```

Output: `checkpoints/KVW_05`.

### 4. Stage A (~30 min)

```bash
bash stage_a.sh 2>&1 | tee logs_stage_a.txt
```

Three sub-steps, run end-to-end:

1. `baselines.KVW --phase stage_a_measure` — per-sample `r_i` + names dumped
   to `kc/r_stageA_forget05.pt`.
2. `eval.py --eval_list forget` — per-sample generation correctness dumped to
   `outputs/stage_a_forget05/eval/forget_per_sample.json`.
3. `scripts/stage_a_analysis.py` — joins by identity name, writes
   `outputs/stage_a_forget05/analysis/auc.json` and the two PNG plots.

---

## Reading the result

```bash
cat outputs/stage_a_forget05/analysis/auc.json
```

Key field: `"auc"`.

- `> 0.70` — strong signal. Send Yejin the bins plot.
- `0.55–0.70` — weak but real. Report with caveats.
- `≈ 0.50` — no signal. The measurement as defined doesn't predict
  recoverability; pivot to representation-level probes.

Plots:
- `stage_a_bins.png` — quintiles of `r_i` vs P(still correct). Monotone up =
  headline figure.
- `stage_a_scatter.png` — sanity visual.

Download to laptop:

```bash
# from laptop
scp -P <pod-ssh-port> root@<pod-ip>:/workspace/KVW/CLEAR/outputs/stage_a_forget05/analysis/*.png ./
```

Or use RunPod's web file browser.

---

## Locked design decisions

Already baked into `stage_a.sh` — don't change without updating the progress
log.

- Measurement timing: **Option B** — on the final post-weakening model
  (matches shipped `num_epochs=1`).
- Layer range in `r_i` sum: `[start_layer, end_layer] = [1, 25]`
  (matches KVW's intervention scope).
- "Correct" label: existing substring-match from `eval.py`.
- Model: Qwen2-VL-2B-Instruct.
- Forget ratio: 5.

---

## Known gotchas

- **`HybridCache` import error from peft**: wrong Python / wrong env. Conda
  env likely isn't active and pip picked up the system `transformers`. Fix
  per the "Env" section above.
- **OOM during finetune at bs=4**: expected on ≤48 GB GPUs. Use bs=1 as
  above.
- **`ValueError: too many values to unpack` in finetune**: upstream bug in
  the grad-accum branch. Fixed on `main` (`f88f957`). Pull latest.
- **`RuntimeError: element 0 ... does not require grad`**: PEFT + grad
  checkpointing. Don't enable `--gradient_accumulation` at all.
- **`data/CLEAR` not found**: symlink or mv as in the Data section.
- **Eval folder name mismatch**: `stage_a.sh` now uses `forget05_perturbed`
  etc., matching `eval.sh`.

---

## Files

```
CLEAR/
  baselines/KVW.py              # + per-sample KC, residual_redundancy, stage_a_measure
  eval.py                       # + per-sample log
  finetune.py                   # grad-accum branch bug fix
  stage_a.sh                    # runner
scripts/
  stage_a_analysis.py           # join by name, AUC + plots
docs/
  idea.md                       # Yejin exchange (source of the question)
  progress_log.md               # running status, decisions, tasks
```
