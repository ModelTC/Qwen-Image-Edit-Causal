import argparse
import math
import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from src.pipeline_qwenimage_edit_plus_causal import QwenImageEditPlusCausalPipeline
from src.transformer_qwenimage_edit_causal import QwenImageTransformerCausal2DModel


def main(
    model_name,
    prompt_list_file: str,
    image_path_list_file: str,
    lora_path: str | None,
    out_dir: str,
    base_seed: int,
    num_inference_steps: int = 8,
    true_cfg_scale: float = 1.0,
):
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    model = QwenImageTransformerCausal2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    )
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = QwenImageEditPlusCausalPipeline.from_pretrained(
        model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
    )
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)
    pipe = pipe.to(device)

    with open(prompt_list_file, "r") as f:
        prompt_list = f.read().splitlines()
    with open(image_path_list_file, "r") as f:
        image_path_list = f.read().splitlines()
    assert len(prompt_list) == len(image_path_list)

    os.makedirs(out_dir, exist_ok=True)

    for i, prompt in enumerate(prompt_list):
        image_paths = image_path_list[i].split(" ")
        input_args = {
            "prompt": prompt,
            "generator": torch.Generator(device=device).manual_seed(base_seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": " ",
            "num_inference_steps": num_inference_steps,
            "image": [
                Image.open(image_path).convert("RGB") for image_path in image_paths
            ],
        }
        image = pipe(**input_args).images[0]
        image.save(
            f"{out_dir}/{i:02d}_{num_inference_steps}steps_cfg{true_cfg_scale}_example.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_list_file", type=str, default="examples/prompt_list.txt"
    )
    parser.add_argument(
        "--image_path_list_file", type=str, default="examples/image_path_list.txt"
    )
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--model_name", type=str, default="Lightx2v/Qwen-Image-Edit-Causal"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1)
    args = parser.parse_args()
    if args.steps is None:
        num_inference_steps = 50 if args.lora_path is None else 8
    else:
        num_inference_steps = args.steps
    if args.cfg is None:
        true_cfg_scale = 4.0 if args.lora_path is None else 1.0
    else:
        true_cfg_scale = args.cfg
    if args.lora_path is not None:
        assert os.path.exists(args.lora_path), (
            f"Lora path {args.lora_path} does not exist"
        )

    main(
        model_name=args.model_name,
        prompt_list_file=args.prompt_list_file,
        image_path_list_file=args.image_path_list_file,
        lora_path=args.lora_path,
        out_dir=args.out_dir,
        base_seed=args.base_seed,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
    )
