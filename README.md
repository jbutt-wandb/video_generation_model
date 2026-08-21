# Video Generation

Generate short cinematic clips **with synchronized audio** from a text prompt — plain
text-to-video, anchored to keyframe images, or grounded in reference photos (put *this*
person in *that* scene). Runs [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
on 2× CoreWeave H100s and logs every run to Weights & Biases. Drive it by hand or let a
coding agent run the whole loop. (A lightweight
[Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) text-to-video script is
included for single-GPU, video-only runs.)

| | |
|---|---|
| Model | MiniMax-H3 (~124 GB bf16: DiT transformer + Qwen3-VL conditioner; each workflow's ~62 GB transformer partition downloads on first use; license-gated) |
| Hardware | 2× H100 80 GB (single GPU works via CPU offload with ~200 GB host RAM) |
| Output | 24 fps mp4 with stereo audio, ~5–15 s (`num_frames` up to ~360) |
| Cluster | CoreWeave Slurm, `h100` partition (no access? contact **Tara Madhyastha** at CoreWeave) |

For a W&B project with example outputs, see: https://wandb.ai/wandb-smle/jb_media_logging

## Setup

1. **Credentials** — `cp .env_example .env` and fill it in: `WANDB_API_KEY`,
   `WANDB_ENTITY`, `WANDB_PROJECT` (videos are logged to *your* project), and `HF_TOKEN`
   (MiniMax-H3 is license-gated — accept on the HF model page first).
2. **Local environment** — `uv sync` (Python 3.10). diffusers is pinned to a git commit;
   do **not** install `kernels` (see [`docs/cluster_environment.md`](docs/cluster_environment.md)).
3. **Cluster mirror** — set up a `coreweave-login` ssh alias (key auth, no password
   prompts — verify with `ssh -o BatchMode=yes coreweave-login true`), then:
   ```bash
   rsync -av --exclude .git --exclude .venv --exclude generated_videos \
     ./ coreweave-login:video_gen_demo/
   ssh coreweave-login "cd ~/video_gen_demo && export PATH=\$HOME/.local/bin:\$PATH && uv sync"
   ```
   Re-run the rsync whenever files change.

## Make a video

Always **prompt guide first, command second** — MiniMax-H3 prompts are structured
labeled-section strings (3 sections for t2va/fl2va, 6 for ref2va), not free text, and
prompt quality is the single biggest lever on output quality. Write the prompt following
[`docs/minimax_prompt_guide.md`](docs/minimax_prompt_guide.md), or crib from the worked,
known-good examples in [`docs/run_commands.md`](docs/run_commands.md).

### The three generation tasks

**`t2va` — text → video + audio.** The whole scene, motion, and soundtrack come from the
prompt alone. The simplest task and the lightest on memory (the full 1344×768 canvas fits).

```bash
uv run scripts/minimax_h3.py --workflow t2va \
  --prompt "A monkey wearing sunglasses swinging on tree branches in the jungle under a starlit canopy, cinematic, 4k" \
  --num_frames 124 --num_inference_steps 30 --seed 42
```

**`fl2va` — keyframe-anchored video + audio.** Pin the video to real images: `--image`
alone makes the clip *start on* that exact frame (great for animating a photo),
`--last_image` alone makes it *end on* one, both together interpolate between them. The
prompt states where each picture lands on the timeline.

```bash
uv run scripts/minimax_h3.py --workflow fl2va \
  --image assets/monkey_start.png --last_image assets/moon_end.png \
  --height 544 --width 960 --num_frames 124 --num_inference_steps 30 --seed 42 \
  --prompt "integrated_multimodal_description: Picture 1 aligns with the 0.00-second mark; Picture 2 aligns with the 5.17-second mark. [Shot 1] ... The camera tilts up with large amplitude at slow speed ... overall_soundscape: ... non_diegetic_music: ..."
```

**`ref2va` — reference-guided video + audio.** Instead of pinning exact frames, you hand
the model reference media it draws from freely — a person's identity from one photo, a
scene from another (max 2 images on this hardware; video/audio refs also supported). Ref
order = `<Picture N>` numbering in the six-section prompt.

```bash
uv run scripts/minimax_h3.py --workflow ref2va \
  --ref assets/monkey_start.png --ref assets/moon_end.png \
  --height 448 --width 768 --num_frames 158 --num_inference_steps 30 --seed 42 \
  --prompt "subject_definitions: <Subject 1> is the monkey shown in <Picture 1> ... summary: [reference generation] ... retention_analysis: ... detailed_description: [Shot 1] ... overall_soundscape: ... non_diegetic_music: ..."
```

The prompts above are abbreviated — full, known-good versions of all three live in
[`docs/run_commands.md`](docs/run_commands.md).

### Running on the cluster

```bash
# Provision an allocation, hop onto the compute node (never run on the login node):
sbatch -p h100 --gres=gpu:2 --cpus-per-task=16 --mem=100G --job-name=videogen --wrap "sleep infinity"
srun --jobid=<JOBID> --overlap --pty bash
cd ~/video_gen_demo
# ...then run any of the task commands above.
```

Release the GPUs with `scancel <JOBID>` when done. Background launches, worked examples
with full prompts, and every discovered constraint: [`docs/run_commands.md`](docs/run_commands.md).

**With a coding agent** (Claude Code opened in this repo) — [`CLAUDE.md`](CLAUDE.md)
teaches the agent the provision → launch → monitor → review → release loop, including the
launch pattern (background `srun` into the allocation, one log file per run) and the log
signals to watch (failure signatures, and the fact that `wandb: View run` prints at
startup — completion is `wandb: Synced` + the saved mp4). You just describe the shot:

```
"Generate a video from assets/monkey_start.png where the monkey waves at the camera.
 Provision the cluster, ground the prompt in the prompting guide."
```

**Iterating:** new take → change `--seed` only; better direction → change the prompt
with the seed fixed (clean A/B, the full config is on every W&B run); expression or
action notes work best directed *chronologically* in the prompt ("his smile relaxes
into…") rather than as scene adjectives.

**Sizing runs (avoid OOM):** on 2× 80 GB, fl2va/ref2va are bounded by roughly
**53M pixels × frames** — e.g. 960×544 or 704×704 @ 124 frames, 768×448 or 480×640
@ 158 frames; the full 1344×768 canvas OOMs. ref2va accepts at most 2 image refs.

## Results

Every run is a W&B run: the video plays in the browser under **Media → "Generated
Video"** (MiniMax-H3's audio track survives the mux and plays too), with the full config
(prompt, seed, canvas, frames, workflow) attached, so any output can be traced and
reproduced. The mp4 also lands on the cluster in `generated_videos/MiniMax-H3/`.

## Learn more

**Read these to understand how the models work and how to make them work** — the
parameters, prompt formats, and hardware constraints are not guessable, and every doc
exists because getting one of them wrong cost a run:

- [`docs/video_generation_guide.md`](docs/video_generation_guide.md) — every parameter of
  both scripts (`num_frames` rules, workflows, seeds, W&B logging)
- [`docs/minimax_prompt_guide.md`](docs/minimax_prompt_guide.md) — the MiniMax-H3 prompt
  formats (read before writing any prompt)
- [`docs/run_commands.md`](docs/run_commands.md) — copy-paste commands, worked cluster
  runs with real prompts, memory ceilings
- [`docs/cluster_environment.md`](docs/cluster_environment.md) — cluster setup and
  error→fix table
- [`docs/gpu_memory_tutorial.md`](docs/gpu_memory_tutorial.md) — tutorial: how a 124 GB
  model runs on 80 GB cards (placement, sharding, offloading, quantization)
- [`CLAUDE.md`](CLAUDE.md) — rules for coding agents driving the cluster workflow

## Layout

```
scripts/            minimax_h3.py (video+audio, 3 workflows) · wan_text_to_video.py
                    (video only, 1 GPU) · check_device.py sanity check
docs/               guides, prompt formats, run commands, GPU memory tutorial
assets/             reference/keyframe images for fl2va and ref2va
generated_videos/   outputs, one subdirectory per model
CLAUDE.md           rules for coding agents driving the cluster workflow
```
