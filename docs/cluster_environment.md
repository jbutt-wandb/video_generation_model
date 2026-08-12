# Cluster Environment Setup — MiniMax-H3 on CoreWeave H100s

How to configure the environment for `minimax_h3.py` (MiniMax-H3 text-to-video-audio
via diffusers Modular Pipelines) so it runs cleanly on the Slurm cluster. Every item below
corresponds to a real failure we hit on 2026-08-10; the fixes are listed in the order the
errors appear if you skip them.

## TL;DR checklist

```bash
# On the cluster, in the project directory (~/video_gen_demo):
export PATH="$HOME/.local/bin:$PATH"     # uv is not on PATH in non-interactive shells
uv add torchvision av                    # required runtime deps missing from a minimal install
# do NOT `uv add kernels` (see below)

# In .env (or the job script):
TORCH_DISABLE_NATIVE_JIT=1               # required on these compute nodes for torch >= 2.13
HF_TOKEN=hf_...                          # newer huggingface_hub ignores HUGGINGFACEHUB_API_TOKEN
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=...
```

Launch pattern that works (from a login node, into an existing allocation):

```bash
cd ~/video_gen_demo
nohup env TORCH_DISABLE_NATIVE_JIT=1 \
  srun --jobid=<JOBID> --overlap --export=ALL \
  .venv/bin/python minimax_h3.py --prompt "..." --num_frames 124 \
  --num_inference_steps 30 --seed 42 > minimax_run.log 2>&1 &
```

Hardware: the two-GPU split path needs 2× H100 80GB (the model is ~124 GB in bf16:
61.7 GB transformer + 62.1 GB Qwen3-VL text encoder). Single-GPU CPU-offload needs
~200 GB host RAM instead.

## The errors and their fixes

### 1. `execve(): uv: No such file or directory`

**Cause:** `uv` lives in `~/.local/bin`, which is only added to PATH by the interactive
shell profile. Non-interactive shells (`ssh host "cmd"`, `srun` payloads, sbatch scripts)
don't source it.

**Fix:** either `export PATH="$HOME/.local/bin:$PATH"` at the top of job scripts, or skip
`uv run` entirely and invoke the venv interpreter directly: `.venv/bin/python script.py`.
The direct interpreter path is the more robust choice inside `srun`.

### 2. `AttributeError: 'NoneType' object has no attribute 'create_mm_token_type_ids'`

**Cause:** missing **torchvision**. The Qwen3-VL processor bundles a video processor that
imports torchvision. Without it, diffusers' `load_components()` fails to build the
`processor` component but only logs a *warning* ("Failed to create component processor")
and leaves it `None` — the crash surfaces much later, in the text-encoder block, with this
misleading error.

**Fix:** `uv add torchvision`. It must match the installed torch CUDA build; resolving it
in-project gives the right wheel (e.g. `torchvision 0.28.0+cu130` for `torch 2.13.0+cu130`).

**Lesson:** if a modular-pipeline component is unexpectedly `None`, search the log for
"Failed to create component" — the real cause is in the warning, not the traceback.

### 3. `subprocess.CalledProcessError: ['/usr/bin/gcc', '.../cuda_utils.c', ...]` (triton)

**Cause:** torch 2.13's `torch._native` op system intercepts certain aten ops (we hit it
via `aten::bmm` in Qwen3-VL's RoPE path) and routes them to triton-JIT kernels. Triton's
JIT gcc-compiles a small C stub against `Python.h` — and the compute nodes have no Python
dev headers (`/usr/include/python3.10/Python.h` doesn't exist; system python, no root to
install `python3-dev`).

**Fix:** set `TORCH_DISABLE_NATIVE_JIT=1` in the environment. This is torch's official
escape hatch (see `torch/_native/common_utils.py`): it skips registering the triton/cuteDSL
native ops so torch uses its standard precompiled CUDA kernels. Results are identical;
the perf difference is negligible for this workload.

This applies to **any** torch ≥ 2.13 workload on these nodes, not just MiniMax-H3 —
put it in the job template.

### 4. Same gcc/triton error via the `kernels` package

**Cause:** installing `kernels` (to enable the Flash Attention 3 backend,
`pipe.transformer.set_attention_backend("_flash_3_hub")`) makes transformers pull
triton-JIT hub kernels, which fail to compile for the same missing-headers reason.

**Fix:** don't install `kernels` on this cluster. The script's `try/except` around
`set_attention_backend` falls back to stock SDPA attention — slower (FA3 is ~3× on
attention) but correct. Revisit if the node images ever gain `python3-dev`.

### 5. `ImportError: PyAV is required to use encode_video`

**Cause:** diffusers' `encode_video` (which muxes the generated video frames and audio
track into one mp4) requires **PyAV** — `imageio-ffmpeg` is not a substitute, and the
compute nodes have no system `ffmpeg`.

**Fix:** `uv add av`.

This one is the most expensive to hit: it throws *after* the full generation (all
denoising steps + VAE decode), so the ~15 minutes of GPU work is lost. Verify
`python -c "import av"` succeeds before launching.

### 6. (Warning, not fatal) unauthenticated Hugging Face Hub requests

**Cause:** the `.env` sets `HUGGINGFACEHUB_API_TOKEN`, a legacy variable name that newer
`huggingface_hub` releases no longer read.

**Fix:** add `HF_TOKEN=<same value>` to `.env`. Harmless while all weights are already in
`~/.cache/huggingface` (the MiniMax-H3 `t2va` partition is ~130 GB), but you'll want it for
fresh downloads (rate limits) and the license-gated repo.

## Failure-mode summary

| Symptom | Root cause | Fix |
|---|---|---|
| `uv: No such file or directory` | `~/.local/bin` not on non-interactive PATH | Use `.venv/bin/python` directly |
| `NoneType ... create_mm_token_type_ids` | torchvision missing → processor silently `None` | `uv add torchvision` |
| gcc fails compiling `cuda_utils.c` | triton JIT needs `Python.h`; nodes lack dev headers | `TORCH_DISABLE_NATIVE_JIT=1`; don't install `kernels` |
| `PyAV is required to use encode_video` | `av` missing (thrown *after* generation) | `uv add av` |
| "unauthenticated requests to the HF Hub" | legacy `HUGGINGFACEHUB_API_TOKEN` var | add `HF_TOKEN` to `.env` |
