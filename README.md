# Video Generation

Text-to-video generation with diffusers, logged to Weights & Biases, designed to be
driven end-to-end by a coding agent (provision → launch → monitor → review). Two models:

| Model | Script | Output |
|---|---|---|
| [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | `scripts/wan_text_to_video.py` | video (1.3B, fits one GPU) |
| [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | `scripts/minimax_h3.py` | video + stereo audio (33B, needs 2× H100 80GB) |

MiniMax-H3 supports three workflows via `--workflow`: `t2va` (text→video+audio),
`fl2va` (first/last keyframe anchored), `ref2va` (reference images/video/audio).
Outputs land in `generated_videos/{model_name}/` and are logged to W&B as a
`wandb.Video` panel (audio track included).

## Requirements

- **Compute:** MiniMax-H3's two-GPU split needs 2× H100 80GB (single GPU works via CPU
  offload with ~200 GB host RAM). It is best to have access to the **CoreWeave training
  cluster** (Slurm, `h100` partition), which this repo's workflow is built around — if
  you don't have access, contact **Tara Madhyastha at CoreWeave**.
- **Credentials** in `.env` (copy `.env_example`): `WANDB_API_KEY`, `WANDB_ENTITY`,
  `WANDB_PROJECT`, `HF_TOKEN` (MiniMax-H3 is license-gated — accept on the HF page first).
- **Environment:** Python 3.10 managed with `uv` (`uv sync`), plus `torchvision` and `av`;
  do NOT install `kernels`. On the cluster, `TORCH_DISABLE_NATIVE_JIT=1` is mandatory.
  Details and error→fix table: [`docs/cluster_environment.md`](docs/cluster_environment.md).

## Quick start (interactive, on a compute node)

```bash
cd ~/video_gen_demo

# MiniMax-H3: text -> video + audio (~15 min on 2x H100)
uv run scripts/minimax_h3.py \
  --prompt "A monkey wearing sunglasses swinging on tree branches in the jungle, cinematic, 4k" \
  --num_frames 124 --num_inference_steps 30 --seed 42

# Wan2.1: text -> video
uv run scripts/wan_text_to_video.py \
  --prompt "A futuristic cityscape at sunset, cinematic, 4k" \
  --num_frames 81 --guidance_scale 5.0 --seed 42
```

## W&B integration

Every run is a W&B run — no extra flags needed. The scripts log to the entity/project
set by `WANDB_ENTITY`/`WANDB_PROJECT` in `.env`, and the run URL is printed at startup
(`wandb: 🚀 View run at ...`) and captured in the run log. On the run page you'll find:

- **Your video** under **Media** in a panel named **"Generated Video"**, uploaded
  automatically when generation finishes — for MiniMax-H3 the audio track survives the
  mux and plays right in the browser.
- The full CLI config (prompt, seed, resolution, frames, workflow) in the run's
  **Overview → Config**, so any output can be traced back to its exact settings and
  reproduced.

This makes W&B the place to review and compare takes across seeds/prompts; the mp4 on
the cluster (`generated_videos/{model_name}/`) is the same file if you prefer scp.

## Running remotely with a coding agent

The intended workflow: clone this repo locally, mirror it to the cluster, and let
your coding agent (e.g. Claude Code) drive runs from your machine over ssh.

One-time setup — an ssh alias the agent can use non-interactively (key auth, no
prompts), then mirror the repo and build the venv:

```bash
# ~/.ssh/config — ask your cluster admin for the login hostname
Host coreweave-login
    HostName <your-cluster-login-hostname>
    User <your-cluster-username>
    IdentityFile ~/.ssh/<your-key>

# Mirror the repo to the cluster home and build the venv there
rsync -av --exclude .git --exclude .venv --exclude generated_videos \
  ./ coreweave-login:video_gen_demo/
ssh coreweave-login "cd ~/video_gen_demo && export PATH=\$HOME/.local/bin:\$PATH && uv sync"
```

(Verify with `ssh -o BatchMode=yes coreweave-login true` — the agent needs this to
succeed without a password prompt. Re-run the rsync whenever scripts/docs change.)

`CLAUDE.md` encodes the rules for the agent; the loop is:

1. **Provision** an allocation — check `sinfo -p h100` for idle nodes and
   `squeue -u $USER` for an existing allocation first:
   ```bash
   sbatch -p h100 --gres=gpu:2 --cpus-per-task=16 --mem=100G --job-name=videogen --wrap "sleep infinity"
   ```
2. **Write the prompt** following [`docs/minimax_prompt_guide.md`](docs/minimax_prompt_guide.md)
   — MiniMax-H3 prompts are structured labeled-section strings, not free text.
3. **Launch** into the allocation from the login node (compute never runs on the
   login node itself), one log file per run:
   ```bash
   cd ~/video_gen_demo && mkdir -p logs
   nohup env TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     srun --jobid=<JOBID> --overlap --export=ALL \
     .venv/bin/python scripts/minimax_h3.py <args> > logs/<name>.log 2>&1 &
   ```
4. **Monitor** the log until a terminal state — progress is a tqdm bar; failure
   signatures are `Traceback` / `CUDA out of memory` / `srun: error` / the PID dying;
   completion is `wandb: Synced` + `Saved video locally to ...mp4`. (`wandb: View run
   at <url>` prints at startup — it is not completion.) An agent should poll the log
   over ssh and alert on all of these, not just success.
5. **Review** the mp4 on the W&B run page (audio plays there) or scp it back to
   local `generated_videos/`, then iterate: fix the seed to A/B prompts, vary the
   seed for new takes.
6. **Release** the allocation when done: `scancel <JOBID>`.

**Sizing runs (avoid OOM):** on 2× 80GB, fl2va/ref2va are bounded by roughly
**53M pixels × frames** — e.g. 960×544 or 704×704 @ 124 frames, 768×448 @ 158 frames;
the full 1344×768 canvas OOMs. ref2va accepts at most 2 image refs on this split.
Known-good commands, worked prompts, and the full constraint list:
[`docs/run_commands.md`](docs/run_commands.md).

## Docs

**Read these to understand how the models work and how to make them work** — the
parameters, prompt formats, and hardware constraints are not guessable, and every
doc exists because getting one of them wrong cost a run:

- [`docs/video_generation_guide.md`](docs/video_generation_guide.md) — every parameter
  of both scripts (`num_frames` rules, workflows, seeds, W&B logging)
- [`docs/minimax_prompt_guide.md`](docs/minimax_prompt_guide.md) — the MiniMax-H3
  prompt formats (read before writing any prompt)
- [`docs/run_commands.md`](docs/run_commands.md) — copy-paste launch commands, worked
  examples with real prompts, memory ceilings
- [`docs/cluster_environment.md`](docs/cluster_environment.md) — environment setup and
  error→fix reference for the compute nodes

## Layout

```
scripts/            generation scripts (+ check_device.py sanity check)
docs/               guides and run commands
assets/             reference images for fl2va/ref2va
generated_videos/   outputs, one subdirectory per model
CLAUDE.md           rules for coding agents driving the cluster workflow
```
