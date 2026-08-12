# Video Generation Guide — Wan2.1 & MiniMax-H3

How to query the two text-to-video models in this repo, what every parameter means, and
how the two models differ. Both scripts share the same skeleton: parse args → load the
diffusers pipeline → generate → export mp4 → log the video to Weights & Biases.

| | Wan2.1-T2V-1.3B | MiniMax-H3 |
|---|---|---|
| Script (local repo) | `scripts/wan_text_to_video.py` | `scripts/minimax_h3.py` |
| Script (cluster) | `~/video_gen_demo/wan_text_to_video.py` | `~/video_gen_demo/minimax_h3.py` |
| Output | video only | video **+ synchronized stereo audio** |
| Size / hardware | 1.3B — fits one GPU easily | 33B + 62GB text encoder — needs 2× H100 80GB (or 1 GPU + ~200GB RAM offload) |
| diffusers API | classic `WanPipeline` | Modular Pipelines (`ModularPipeline`, experimental) |
| Guidance | classifier-free guidance (`guidance_scale`, `negative_prompt`) | **guidance-distilled** — no CFG knobs at all |
| Native fps | 15–16 | 24 (fixed) |
| Frame-count rule | `4n + 1` (81 ✓) | `17n + 5` (124 ✓) |
| Typical latency | minutes on one H100 | ~15 min for 124 frames on 2× H100 (no FA3) |

## Querying Wan2.1

```bash
uv run scripts/wan_text_to_video.py \
  --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --prompt "A futuristic cityscape at sunset, with flying cars and neon lights, cinematic, 4k" \
  --negative_prompt "blurry, low quality, static, deformed, oversaturated" \
  --height 480 --width 832 \
  --num_frames 81 --fps 15 \
  --guidance_scale 5.0 --seed 42
```

## Querying MiniMax-H3

```bash
# Cluster (see CLUSTER_ENVIRONMENT.md for the required env setup):
cd ~/video_gen_demo
nohup env TORCH_DISABLE_NATIVE_JIT=1 \
  srun --jobid=<JOBID> --overlap --export=ALL \
  .venv/bin/python minimax_h3.py \
  --prompt "A monkey wearing sunglasses swinging on tree branches in the jungle, cinematic, 4k" \
  --num_frames 124 --num_inference_steps 30 --seed 42 \
  > minimax_run.log 2>&1 &
```

The script picks its GPU strategy automatically: with 2+ visible GPUs it splits the model
(text encoder on `cuda:1`, transformer + VAEs on `cuda:0`); with 1 GPU it streams
components through CPU offload.

## Parameter reference

### Shared parameters

**`--prompt`** — the text description of the video. Both models reward specificity:
subject, action, setting, camera/style qualifiers ("cinematic", "4k"). MiniMax-H3 also
uses the prompt to shape the *audio* (mention sounds explicitly: "snow crunching
underfoot", "jungle ambience").

**`--height` / `--width`** — output resolution in pixels.
- *Wan2.1*: the 1.3B checkpoint is trained at 480p; keep the default **480×832** for
  best quality. Other sizes work but degrade quickly away from the training resolution.
- *MiniMax-H3*: trained with a **768-pixel short edge**; both dimensions must be
  **multiples of 32**. Default 1344×768 (16:9). A smaller canvas like 960×544 is ~2.3×
  faster per step if you're iterating on prompts. (2K output exists upstream via a
  separate regeneration module that is *not* in the diffusers integration.)

**`--num_frames`** — how many frames to generate; combined with fps this sets the clip
duration. Video VAEs compress time in fixed-size chunks, so frame counts must line up
with the temporal compression window:
- *Wan2.1*: must be **4n + 1** (…, 49, 81, 121, …). Default 81 → 81/15 ≈ 5.4 s.
- *MiniMax-H3*: snapped **up** to the next **17n + 5** (…, 90, 107, 124, …). Default
  124 → 124/24 ≈ 5.2 s. Supported range is 5–15 seconds (~124 to ~360 frames).
- More frames = proportionally more GPU time and memory.

**`--fps`** — playback frame rate written into the mp4 (and passed to `wandb.Video`).
It does **not** change what the model generates — the model produces a fixed number of
frames; fps just decides how fast they play. Keep it at the model's native rate
(Wan ≈ 15–16, MiniMax-H3 = 24) or motion will look sped-up/slowed-down.

**`--seed`** — initializes the random noise the video is denoised from. Same
seed + same parameters = same video (bit-identical on the same hardware/software).
Change the seed to get a different take on the same prompt; fix it to compare
prompt/parameter tweaks apples-to-apples.

**`--model_id`** — the Hugging Face repo to load (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` /
`MiniMaxAI/MiniMax-H3`). Only change it for a different checkpoint variant of the same
architecture.

### Wan2.1-only parameters

**`--negative_prompt`** — text describing what to *avoid* ("blurry, low quality,
deformed…"). Works through classifier-free guidance: each step compares the prompt-
conditioned prediction against the negative-conditioned one and pushes away from the
latter.

**`--guidance_scale`** — how hard to push toward the prompt (CFG weight). Default 5.0.
Lower (≈3–4) = more natural motion, looser prompt adherence; higher (≈6–8) = stricter
adherence but risks oversaturated colors and stiff, "burnt" motion. Every step runs two
forward passes (conditional + unconditional), which is why CFG models cost ~2× per step.

### MiniMax-H3-only parameters

**`--num_inference_steps`** — how many denoising steps to run (default 30). More steps =
more detail/coherence up to diminishing returns (~40–50); fewer (~20) is faster and fine
for drafts. Runtime scales linearly with this.
(Wan2.1's script doesn't expose this; its pipeline default is 50.)

**`--workflow`** — which task variant to load; also controls which of the two
~62 GB transformer partitions downloads on first use:
| Workflow | Task | Extra CLI inputs |
|---|---|---|
| `t2va` (default) | text → video + audio | none |
| `fl2va` | keyframe-anchored → video + audio | `--image` (start frame), `--last_image` (end frame), or both |
| `ref2va` | reference-guided → video + audio | `--ref <path>`, repeated once per file |

**`--image` / `--last_image`** (fl2va) — paths to the keyframes. The start frame is
stretched onto the canvas (omit `--height/--width` to derive the canvas from its
aspect ratio); the end frame is cover-cropped, and can be passed alone to generate
a video that ends *on* that frame.

**`--ref`** (ref2va) — path to one reference file, repeated per file; the type is
inferred from the extension. **Order matters**: the model reads references in the
order given, so "the person in the first image" refers to your first `--ref`.
Limits: ≤9 images, ≤3 video clips (2–15 s each), ≤3 audio clips (2–15 s each),
≤12 files total, and audio cannot be the only kind. Reference images are encoded
at a 2048-px short edge, so high-res inputs help. (Resampling reference audio that
isn't already at the audio VAE's sample rate requires `torchaudio`.)

**Why no `negative_prompt` / `guidance_scale`?** The released checkpoint is
**guidance-distilled**: the effect of CFG was baked into the weights during training.
Every step is a single forward pass and there is nothing to tune — passing these
parameters isn't supported at all.

### W&B logging (both scripts)

Runs log to the entity/project from `WANDB_ENTITY` / `WANDB_PROJECT` env vars (cluster
`.env` points at `wandb-smle/jb_media_logging`). The full CLI config is stored on the run,
and the finished mp4 is logged as a `wandb.Video` panel named "Generated Video" — for
MiniMax-H3 the audio track survives the mux and plays in the W&B UI. Each script also
contains a commented-out "Method B" block showing how to log generations into a
`wandb.Table` (prompt / seed / resolution / video per row) for side-by-side comparison
across runs.

## Practical recipes

- **Iterating on a prompt (MiniMax-H3):** `--num_inference_steps 20`, 960×544 canvas,
  fixed seed → ~3–4× faster drafts; restore 30 steps / 1344×768 for the final render.
- **Different takes of one prompt:** keep everything fixed, vary `--seed`.
- **A/B testing prompts:** fix the seed, change only the prompt, compare the two W&B runs.
- **Longer clips (MiniMax-H3):** raise `--num_frames` in 17-frame increments
  (141, 158, …, up to ~360 ≈ 15 s); expect proportionally longer runtimes.
