# Minimax H3

## Text -> video + audio (t2va)
### Full
uv run ./scripts/minimax_h3.py --model_id "MiniMaxAI/MiniMax-H3" \
    --workflow "t2va" \
    --prompt "A monkey wearing sunglasses swingling on tree branches in the jungle under a starlit canopy, cinematic, 4k" \
    --height 768 \
    --width 1344 \
    --num_frames 124 \
    --num_inference_steps 30 \
    --fps 24 \
    --seed 42

### Partial
uv run ./scripts/minimax_h3.py \
    --prompt "A monkey wearing sunglasses swingling on tree branches in the jungle under a starlit canopy, cinematic, 4k" \
    --num_frames 124 \
    --num_inference_steps 30 \
    --seed 42

## First frame -> video + audio (fl2va)
Animates from a supplied start image. `--image` is the keyframe the video starts
from (stretched onto the canvas; omit `--height/--width` to keep its aspect ratio).

> NOTE: each workflow downloads its own transformer partition (~62 GB) on first run.

uv run ./scripts/minimax_h3.py \
    --workflow "fl2va" \
    --prompt "The monkey leaps to a higher branch as the camera follows, jungle ambience, cinematic, 4k" \
    --image assets/monkey_start.png \
    --num_frames 124 \
    --num_inference_steps 30 \
    --seed 42

## First + last frame -> video + audio (fl2va with --last_image)
Same `fl2va` workflow: interpolates the motion between two supplied keyframes.
`--last_image` is cover-cropped onto the canvas; it can also be passed *without*
`--image` to generate a video that ends on that frame.

uv run ./scripts/minimax_h3.py \
    --workflow "fl2va" \
    --prompt "The camera tilts up from the monkey in the jungle to the full moon in the night sky, cinematic, 4k" \
    --image assets/monkey_start.png \
    --last_image assets/moon_end.png \
    --num_frames 124 \
    --num_inference_steps 30 \
    --seed 42

## Reference-guided -> video + audio (ref2va)
Conditions on reference media: up to 9 images, 3 video clips (2-15 s each) and
3 audio clips (2-15 s each), max 12 files total (audio cannot be the only kind).
Repeat `--ref` once per file — order matters, the model reads references in the
order given ("the person in the first image..."). `--num_frames` is required.

uv run ./scripts/minimax_h3.py \
    --workflow "ref2va" \
    --prompt "The monkey from the first image dances on a jungle branch under the full moon from the second image, cinematic, 4k" \
    --ref assets/monkey_start.png \
    --ref assets/moon_end.png \
    --num_frames 124 \
    --num_inference_steps 30 \
    --seed 42

# Worked examples (cluster, 2026-08-11)

Real runs on 2x H100 (all logged to W&B `wandb-smle/jb_media_logging`; each
run's full prompt is stored in its config). Launch pattern from the login node
against an existing allocation:

```bash
# Allocation (if none): sbatch -p h100 --gres=gpu:2 --cpus-per-task=16 --mem=100G \
#   --job-name=videogen --wrap "sleep infinity"
cd ~/video_gen_demo && mkdir -p logs
nohup env TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  srun --jobid=<JOBID> --overlap --export=ALL \
  .venv/bin/python scripts/minimax_h3.py <args> > logs/<name>.log 2>&1 &
```

## Memory constraints discovered (80GB cards, 2-GPU split)

- Full 1344x768 canvas **OOMs** for fl2va/ref2va. Safe pixel budget is
  ~520k px: **960x544** (16:9), **704x704** (square), **608x704** (portrait).
  Match the canvas aspect to the keyframe to avoid stretching.
- The ~520k px budget assumes **124 frames** — the real ceiling is
  pixels x frames ~= 53M (608x704 @ 124 thrashes; 960x544 @ 158 = 82M OOMs
  on the denoise GPU at step 1). For 158 frames use **768x448** (54M — ran
  clean at ~8 s/it, 2026-08-12, run 2nr3hj7k).
- ref2va: **max 2 image references** — refs are encoded by the Qwen3-VL
  conditioner at a fixed 2048px short edge, and 3 refs OOM the text-encoder
  GPU. Expect slower steps (~6-8 s/it) with harmless
  `expandable_segments: memory mapping failed` warnings.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required headroom.

## First frame only — animate a still (structure proven in runs zw28uw5j / w6vahc82)

Image held at 0.00s, then a sequence of actions. Match the canvas aspect to the
image (square -> 704x704; portrait -> 448x768 or 608x704). Prompt follows the
guide (see minimax_prompt_guide.md): opens with "At 0.00 seconds into the
target video, Picture 1 is fully referenced."

uv run ./scripts/minimax_h3.py \
    --workflow "fl2va" \
    --image assets/monkey_start.png \
    --height 768 --width 448 \
    --num_frames 124 --num_inference_steps 30 --seed 42 \
    --prompt "integrated_multimodal_description: At 0.00 seconds into the target video, Picture 1 is fully referenced. [Shot 1] Live-action, cinematic. The scene begins exactly as established by Picture 1: a golden-furred monkey wearing blue sunglasses hanging from two vines in a lush jungle. The camera holds steady with small amplitude. The monkey swings gently, lets go of one vine, waves its free hand at the camera, and breaks into an open-mouthed, delighted grin. overall_soundscape: Rustling leaves and creaking vines as the monkey swings, a cheerful chittering call, bright jungle ambience with birdsong. non_diegetic_music: A light, playful marimba motif with an upbeat feel."

## First + last frame — camera move between two stills (run 3o3dz1gg)

Monkey keyframe at 0.00s, moon keyframe at 5.17s (124 frames / 24 fps);
the prompt states both alignments and the camera move as type + amplitude
+ speed.

uv run ./scripts/minimax_h3.py \
    --workflow "fl2va" \
    --image assets/monkey_start.png --last_image assets/moon_end.png \
    --height 544 --width 960 \
    --num_frames 124 --num_inference_steps 30 --seed 42 \
    --prompt "integrated_multimodal_description: Picture 1 aligns with the 0.00-second mark; Picture 2 aligns with the 5.17-second mark. [Shot 1] Live-action, cinematic, night. The scene begins in the position established by Picture 1: a monkey wearing sunglasses perched on a tree branch in a jungle under a starlit canopy. The monkey slowly raises its head and looks up toward the night sky. The camera tilts up and pans away from the monkey with large amplitude at slow speed, rising past silhouetted branches and leaves, and settles on the composition established by Picture 2: a glowing full moon in the dark night sky. overall_soundscape: Soft jungle ambience with insect chirps and a gentle rustle of leaves as the monkey moves; the ambience fades as the camera rises, leaving quiet night air with a light breeze. non_diegetic_music: A gentle, wondrous orchestral swell that builds slowly and peaks as the moon fills the frame."

## Reference-guided — subject identity + scene reference (structure proven in runs wc5b9i90, lvcfy2bp, 2nr3hj7k)

Ground a subject's identity in one reference and pull the scene/backdrop from
another: ref order = Picture numbering, so cite <Picture 1> for the subject and
<Picture 2> for the scene. Six-section ref2va prompt format (see
minimax_prompt_guide.md). For natural gestures, direct the performance as one
relaxed continuous motion rather than a list of beats. Remember the 2-image-ref
limit and the ~53M pixels x frames ceiling (768x448 @ 158 frames is proven).

uv run ./scripts/minimax_h3.py \
    --workflow "ref2va" \
    --ref assets/monkey_start.png --ref assets/moon_end.png \
    --height 448 --width 768 \
    --num_frames 158 --num_inference_steps 30 --seed 42 \
    --prompt "subject_definitions: <Subject 1> is the monkey shown in <Picture 1>: a small golden-furred monkey wearing blue sunglasses. <Picture 2> is the night scene the video is set against: a glowing full moon in a dark sky. summary: [reference generation] The target video shows <Subject 1> perched on a jungle branch at night, gazing up at the full moon from <Picture 2> and hooting at it. retention_analysis: <Subject 1> (appears in [Shot 1]): fully_preserved - golden fur and blue sunglasses retained throughout. <Picture 2> (appears in [Shot 1]): partially_preserved - the full moon and dark night sky serve as the backdrop. detailed_description: [Shot 1] Live-action, cinematic, night. <Subject 1> sits perched on a jungle branch, silhouetted leaves framing the shot, with the glowing full moon from <Picture 2> high in the dark sky behind it. The camera pushes in with small amplitude at slow speed. In one relaxed, continuous motion, <Subject 1> tilts its head up toward the moon, raises both arms, and lets out a long, joyful hoot before settling back onto the branch, still gazing upward. overall_soundscape: Quiet night jungle ambience with crickets and a light breeze; a long clear monkey hoot echoing into the night; a soft rustle of leaves as it settles. non_diegetic_music: A gentle, wondrous orchestral swell that peaks as the monkey calls at the moon."

# Wan
## Full
uv run ./scripts/wan_text_to_video.py --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "A futuristic cityscape at sunset, with flying cars and neon lights, cinematic, 4k" \
    --negative_prompt "blurry, low quality, static, deformed, oversaturated" \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --fps 15 \
    --guidance_scale 5.0 \
    --seed 42 

## Partial
uv run ./scripts/wan_text_to_video.py --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "A monkey wearing sunglasses swingling on tree branches in the jungle under a starlit canopy, cinematic, 4k" \
    --negative_prompt "blurry, low quality, static, deformed, oversaturated"