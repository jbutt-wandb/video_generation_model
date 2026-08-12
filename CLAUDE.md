# CLAUDE.md

Rules for coding agents working in this repo. The GPU workloads run remotely on a
CoreWeave Slurm cluster; the agent drives everything from the local machine over ssh.

## Cluster execution — compute nodes only, never the login node

- Connect with `ssh coreweave-login`. The login node is for orchestration only
  (sbatch/squeue/scancel, file transfer, tailing logs). **Never run Python
  workloads, model downloads, or anything GPU/CPU-heavy on the login node.**
- All generation runs on a compute node, always through an allocation:
  1. `sbatch -p h100 --gres=gpu:2 --cpus-per-task=16 --mem=100G --job-name=videogen --wrap "sleep infinity"`
  2. launch work into it with `srun --jobid=<JOBID> --overlap --export=ALL`
- `uv` is not on PATH in non-interactive/srun shells — invoke `.venv/bin/python`
  directly. Full environment quirks and error→fix table: `docs/cluster_environment.md`.
- Release the allocation (`scancel <JOBID>`) when iteration is done; confirm with
  `squeue`. Don't leave idle GPUs held.

## Provisioning — check resources fit before launching

Before every provision/launch, verify the request is appropriate:

- `sinfo -p h100` — confirm idle nodes exist; `squeue -u $USER` — confirm no
  forgotten allocation is already running (reuse it if so).
- MiniMax-H3 two-GPU split needs 2× H100 80GB; single-GPU CPU offload needs
  ~200 GB host RAM instead. Wan2.1-1.3B fits one GPU.
- Check the memory ceilings in `docs/run_commands.md` before choosing
  `--height/--width/--num_frames`: fl2va/ref2va OOM above roughly
  **53M pixels × frames** (e.g. 768×448 @ 158 frames is the proven landscape
  config; the full 1344×768 canvas OOMs). ref2va takes at most **2 image refs**
  on the 2-GPU split. Always set `TORCH_DISABLE_NATIVE_JIT=1` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Prompting — read the guide first

**Always read `docs/minimax_prompt_guide.md` before composing or sending any
model command.** MiniMax-H3 prompts are structured strings, not free text:
three labeled sections for t2va/fl2va (`integrated_multimodal_description` /
`overall_soundscape` / `non_diegetic_music`), six for ref2va
(`subject_definitions` / `summary` / `retention_analysis` / …), with explicit
keyframe-timestamp alignment and camera moves as type + amplitude + speed.
Worked, known-good prompts live in `docs/run_commands.md`.

## Run, monitor, log

- Launch pattern (from the login node, into the allocation), one log file per run:

  ```bash
  cd ~/video_gen_demo && mkdir -p logs
  nohup env TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    srun --jobid=<JOBID> --overlap --export=ALL \
    .venv/bin/python scripts/minimax_h3.py <args> > logs/<name>.log 2>&1 &
  ```

- Monitor the log until a terminal state, watching for **failure signatures as
  well as success**: `Traceback`, `CUDA out of memory`, `srun: error`,
  `RuntimeError`, `Killed`, or the launched PID disappearing. Completion markers
  are `wandb: Synced` and `Saved video locally to ...mp4` — note that
  `wandb: View run at <url>` prints at *startup* and is not a completion signal.
  `expandable_segments: memory mapping failed` warnings are harmless.
- Every run logs to W&B (entity/project from `.env`); the run URL is in the log
  and the finished mp4 is on the run page. Cluster-side outputs land in
  `generated_videos/MiniMax-H3/`; scp finals back to local `generated_videos/`.
- After a run teaches something new (OOM boundary, working recipe), record it in
  `docs/run_commands.md` and keep `scripts/` and `docs/` in sync between the
  local repo and the cluster mirror (`~/video_gen_demo`).
