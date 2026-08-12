import argparse
import os
import torch
import wandb
import uuid
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from dotenv import load_dotenv

load_dotenv(override=True) 

def load_model(model_id):
    print("Loading VAE and Wan2.1 Pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,  # Keep VAE in float32 to prevent visual artifacts
    )

    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    return pipe


def main(config):
    print("Starting video generation process...")
    # Select hardware device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")

    # Set seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(config["seed"])

    # Load Model
    pipe = load_model(config["model_id"])
    pipe.to(device)

    # Log video to WB
    with wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        config=config,
        name=f"wan2.1-t2v-1.3b-generation-{str(uuid.uuid1())[0:5]}",
    ) as run:

        print(f"Generating video for prompt: '{config['prompt']}'")

        output_frames = pipe(
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            height=config["height"],
            width=config["width"],
            num_frames=config["num_frames"],
            guidance_scale=config["guidance_scale"],
            generator=generator,
        ).frames[0]

        model_name = config["model_id"].split("/")[-1]
        output_dir = os.path.join("generated_videos", model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, "wan_output.mp4")
        export_to_video(
            output_frames, 
            output_filename, 
            fps=config["fps"],
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

    # Example usage:
    """
    uv run wan_text_to_video.py --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "A futuristic cityscape at sunset, with flying cars and neon lights, cinematic, 4k" \
    --negative_prompt "blurry, low quality, static, deformed, oversaturated" \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --fps 15 \
    --guidance_scale 5.0 \
    --seed 42 
    """

    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan2.1.")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="The model ID to use for video generation.")
    parser.add_argument("--prompt", type=str, default="An astronaut riding a horse through a glowing, bioluminescent forest at night, cinematic, 4k", help="The text prompt to generate the video from.")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, static, deformed, oversaturated", help="The negative prompt to avoid certain features in the generated video.")
    parser.add_argument("--height", type=int, default=480, help="The height of the generated video.")
    parser.add_argument("--width", type=int, default=832, help="The width of the generated video.")
    parser.add_argument("--num_frames", type=int, default=81, help="The number of frames in the generated video.")
    parser.add_argument("--fps", type=int, default=15, help="The frames per second of the generated video.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="The guidance scale for the video generation.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed for reproducibility.")

    args = parser.parse_args()

    # Set configuration parameters
    CONFIG = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        }

    # config = {
    # "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    # "prompt": "An astronaut riding a horse through a glowing, bioluminescent forest at night, cinematic, 4k",
    # "negative_prompt": "blurry, low quality, static, deformed, oversaturated",
    # "height": 480,
    # "width": 832,
    # "num_frames": 81,
    # "fps": 15,
    # "guidance_scale": 5.0,
    # "seed": 42,
    # }

    main(config=CONFIG)
