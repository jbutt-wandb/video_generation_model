import argparse
import os

# Must be set before `import torch`: torch 2.13's _native ops otherwise register
# triton-JIT kernels that fail to compile on nodes without Python dev headers
# (see CLUSTER_ENVIRONMENT.md).
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
import wandb
import uuid
from PIL import Image
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.utils.export_utils import encode_video
from dotenv import load_dotenv

load_dotenv(override=True)

# MiniMax-H3 blocks never open media files themselves: the CLI takes paths and
# decodes them here, routed to a reference class by file extension.
REFERENCE_TYPES = {
    ".png": MiniMaxH3ImageReference,
    ".jpg": MiniMaxH3ImageReference,
    ".jpeg": MiniMaxH3ImageReference,
    ".webp": MiniMaxH3ImageReference,
    ".bmp": MiniMaxH3ImageReference,
    ".mp4": MiniMaxH3VideoReference,
    ".mov": MiniMaxH3VideoReference,
    ".mkv": MiniMaxH3VideoReference,
    ".webm": MiniMaxH3VideoReference,
    ".wav": MiniMaxH3AudioReference,
    ".mp3": MiniMaxH3AudioReference,
    ".flac": MiniMaxH3AudioReference,
    ".m4a": MiniMaxH3AudioReference,
    ".ogg": MiniMaxH3AudioReference,
}


def build_media_kwargs(config):
    """Validate the workflow/media combination and decode the media inputs."""
    workflow = config["workflow"]

    if workflow == "t2va":
        if config["image"] or config["last_image"] or config["ref"]:
            raise ValueError(
                "t2va is text-only: use --workflow fl2va for keyframes or --workflow ref2va for references."
            )
        return {}

    if workflow == "fl2va":
        if config["ref"]:
            raise ValueError("--ref belongs to the ref2va workflow.")
        if not (config["image"] or config["last_image"]):
            raise ValueError("fl2va needs --image (start frame), --last_image (end frame) or both.")
        media_kwargs = {}
        if config["image"]:
            media_kwargs["image"] = Image.open(config["image"]).convert("RGB")
        if config["last_image"]:
            media_kwargs["last_image"] = Image.open(config["last_image"]).convert("RGB")
        return media_kwargs

    # ref2va
    if config["image"] or config["last_image"]:
        raise ValueError("--image/--last_image belong to the fl2va workflow.")
    if not config["ref"]:
        raise ValueError("ref2va needs at least one --ref (repeat the flag once per file; order matters).")
    references = []
    for path in config["ref"]:
        ext = os.path.splitext(path)[1].lower()
        if ext not in REFERENCE_TYPES:
            raise ValueError(
                f"Unsupported reference extension {ext!r} ({path}); "
                f"supported: {', '.join(sorted(REFERENCE_TYPES))}."
            )
        references.append(REFERENCE_TYPES[ext].from_file(path))
    return {"references": references}


def load_model(model_id, workflow, num_gpus):
    # MiniMax-H3 weighs ~124GB in bf16 (61.7GB transformer + 62.1GB Qwen3-VL
    # text encoder), so it cannot sit on a single 80GB H100 without offloading.
    # Each workflow loads its own transformer partition (`transformer/` for
    # t2va/fl2va, `transformer_ref/` for ref2va), downloaded on first use.
    if num_gpus >= 2:
        # Two-card split: conditioner on cuda:1, transformer/VAEs on cuda:0
        print(f"Loading MiniMax-H3 Modular Pipeline (workflow={workflow}) split across 2 GPUs...")
        blocks = ModularPipeline.from_pretrained(model_id).blocks.get_workflow(workflow)

        # The Qwen3-VL conditioner encodes MiniMax-H3's *presentation* of the
        # request — for fl2va/ref2va that includes the prepared keyframes or
        # references — so the media-preparation step must ride along with the
        # text encoder. (t2va has no before_encode block; the dict then holds
        # just the text encoder, as before.)
        conditioner_blocks = SequentialPipelineBlocks.from_blocks_dict(
            {
                name: blocks.sub_blocks.pop(name)
                for name in ("before_encode", "text_encoder")
                if name in blocks.sub_blocks
            }
        )

        text_manager = ComponentsManager()
        text_manager.enable_auto_cpu_offload(device="cuda:1")
        conditioner = conditioner_blocks.init_pipeline(model_id, components_manager=text_manager)
        conditioner.load_components(dtype=torch.bfloat16)

        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda:0")
        pipe = blocks.init_pipeline(model_id, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
    else:
        # Single 80GB card: auto CPU offload shuttles components between host RAM
        # and the GPU as each stage runs (needs ~150GB free host RAM)
        print(f"Loading MiniMax-H3 Modular Pipeline (workflow={workflow}) on 1 GPU with CPU offload...")
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(model_id, components_manager=manager)
        pipe.load_components(workflow=workflow, dtype=torch.bfloat16)
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
        conditioner = None

    try:
        # Flash Attention 3 on Hopper (H100): ~3x faster attention.
        # ref2va denoises with the `transformer_ref` partition instead.
        transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_ref", None)
        transformer.set_attention_backend("_flash_3_hub")
    except Exception as e:
        print(f"Flash Attention 3 unavailable, using default backend ({e})")

    return conditioner, pipe


def main(config):
    print("Starting video generation process...")
    # Validate and decode the media inputs first: fail fast on a bad request
    # before spending minutes loading 124GB of weights.
    media_kwargs = build_media_kwargs(config)

    # Select hardware strategy
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("MiniMax-H3 needs at least one 80GB-class CUDA GPU (124GB of bf16 weights).")

    print(f"Using {num_gpus} CUDA GPU(s)")

    # Set seed for reproducibility (CPU generator works across both GPU layouts)
    generator = torch.Generator().manual_seed(config["seed"])

    # Load Model
    conditioner, pipe = load_model(config["model_id"], config["workflow"], num_gpus)

    # Log video to WB
    with wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        config=config,
        name=f"minimax-h3-{config['workflow']}-generation-{str(uuid.uuid1())[0:5]}",
    ) as run:

        print(f"Generating video for prompt: '{config['prompt']}' (workflow={config['workflow']})")

        # MiniMax-H3 is guidance-distilled: no negative_prompt / guidance_scale
        gen_kwargs = dict(
            num_frames=config["num_frames"],
            num_inference_steps=config["num_inference_steps"],
            generator=generator,
            output=["videos", "audio", "sampling_rate"],
        )
        # height/width are optional: left unset, the pipeline resolves the
        # canvas itself (16:9 for t2va, the keyframe's aspect ratio for fl2va)
        size_kwargs = {}
        if config["height"] is not None:
            size_kwargs = {"height": config["height"], "width": config["width"]}

        if conditioner is not None:
            # Two-GPU path: media preparation + prompt encoding on cuda:1,
            # denoise/decode on cuda:0. The media — and the geometry the
            # preparation step resolves (the canvas; for ref2va also
            # num_frames) — go to the conditioner; the resolved values reach
            # the main pipe inside `state`.
            cond_kwargs = dict(prompt=config["prompt"], **media_kwargs)
            if config["workflow"] == "t2va":
                gen_kwargs.update(size_kwargs)
            else:
                cond_kwargs.update(size_kwargs)
                if config["workflow"] == "ref2va":
                    cond_kwargs["num_frames"] = gen_kwargs.pop("num_frames")
            state = conditioner(**cond_kwargs)
            results = pipe(state=state, **gen_kwargs)
        else:
            results = pipe(prompt=config["prompt"], **media_kwargs, **size_kwargs, **gen_kwargs)

        model_name = config["model_id"].split("/")[-1]
        output_dir = os.path.join("generated_videos", model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"minimax_h3_{config['workflow']}_output.mp4")
        # Video and audio come back separately; encode_video muxes them into one mp4
        encode_video(
            results["videos"][0],
            fps=config["fps"],
            output_path=output_filename,
            audio=results["audio"][0],
            audio_sample_rate=results["sampling_rate"],
        )

        print(f"Saved video locally to {output_filename}")

        video = wandb.Video(
                output_filename,
                caption=config["prompt"],
                fps=config["fps"],
                format="mp4",
            )

        # Method A: Direct Video Logging
        panel_name = "Generated Video"
        run.log({panel_name: video})

        # Method B: Log to a W&B Table
        # table = wandb.Table(columns=["Prompt", "Seed", "Resolution", "Frames", "Video"])
        # table.add_data(
        #     config["prompt"],
        #     config["seed"],
        #     f"{config['width']}x{config['height']}",
        #     config["num_frames"],
        #     wandb.Video(output_filename, fps=config["fps"], format="mp4")
        # )

        # run.log({"generation_results_table": table})

    # Run automatically finishes when exiting the 'with' block (even if an error occurs)
    print(f"Successfully finished run={run.id} and logged generation results to Weights & Biases!")

if __name__ == "__main__":

    # Example usage (Slurm, 2x H100):
    """
    srun --partition=h100 --gres=gpu:2 --cpus-per-task=16 --mem=200G \
    uv run minimax_h3.py --model_id "MiniMaxAI/MiniMax-H3" \
    --workflow "t2va" \
    --prompt "A monkey wearing sunglasses swingling on tree branches in the jungle under a starlit canopy, cinematic, 4k" \
    --height 768 \
    --width 1344 \
    --num_frames 124 \
    --num_inference_steps 30 \
    --fps 24 \
    --seed 42

    # First/last keyframe (fl2va): pass --image, --last_image or both
    ... --workflow "fl2va" --image start.png --last_image end.png ...

    # Reference-guided (ref2va): repeat --ref once per file, order matters
    ... --workflow "ref2va" --ref person.jpg --ref song.wav ...
    """

    parser = argparse.ArgumentParser(description="Generate a video with synchronized audio from a text prompt using MiniMax-H3 (workflows: t2va = text only, fl2va = first/last keyframe, ref2va = reference-guided).")
    parser.add_argument("--model_id", type=str, default="MiniMaxAI/MiniMax-H3", help="The model ID to use for video generation.")
    parser.add_argument("--workflow", type=str, default="t2va", choices=["t2va", "fl2va", "ref2va"], help="The MiniMax-H3 workflow to load; each downloads its own ~62GB transformer partition on first use.")
    parser.add_argument("--prompt", type=str, default="An astronaut riding a horse through a glowing, bioluminescent forest at night, cinematic, 4k", help="The text prompt to generate the video from.")
    parser.add_argument("--image", type=str, default=None, help="fl2va: path to the keyframe the video starts from (stretched onto the canvas).")
    parser.add_argument("--last_image", type=str, default=None, help="fl2va: path to the keyframe the video ends on; may be passed without --image to generate a video ending on that frame.")
    parser.add_argument("--ref", action="append", default=None, metavar="PATH", help="ref2va: reference media file (image/video/audio by extension), repeated once per file in the order the model should read them. Limits: 9 images, 3 videos (2-15s each), 3 audio clips (2-15s each), 12 files total; audio cannot be the only kind.")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video (multiple of 32, short edge 768). Passed together with --width, or omit both to let the model pick the canvas.")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video (multiple of 32). Passed together with --height.")
    parser.add_argument("--num_frames", type=int, default=124, help="The number of frames in the generated video (snapped up to 17*n + 5; 5-15s at 24 fps).")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="The number of denoising steps.")
    parser.add_argument("--fps", type=int, default=24, help="The frames per second of the generated video (MiniMax-H3 generates at 24 fps).")
    parser.add_argument("--seed", type=int, default=42, help="The random seed for reproducibility.")

    args = parser.parse_args()

    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be passed together.")

    # Set configuration parameters
    CONFIG = {
        "model_id": args.model_id,
        "workflow": args.workflow,
        "prompt": args.prompt,
        "image": args.image,
        "last_image": args.last_image,
        "ref": args.ref,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "fps": args.fps,
        "seed": args.seed,
        }

    main(config=CONFIG)
