import argparse
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from src.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipelineForBench
from src.pipeline_qwenimage_edit_plus_causal import QwenImageEditPlusCausalPipeline
from src.transformer_qwenimage import QwenImageTransformer2DModelFixSpeed
from src.transformer_qwenimage_edit_causal import QwenImageTransformerCausal2DModel


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Linear(100, 100)  # on `cuda` device
    ```
    """

    if include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter  # pyright: ignore
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def main(
    model_name: str,
    is_causal: bool = True,
    num_inference_steps: int = 8,
    true_cfg_scale: float = 1.0,
):
    torch_dtype = torch.bfloat16
    device = "cuda"

    print("is causal", is_causal)
    config = {
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "guidance_embeds": False,
        "in_channels": 64,
        "joint_attention_dim": 3584,
        "num_attention_heads": 24,
        "num_layers": 60,
        "out_channels": 16,
        "patch_size": 2,
        "zero_cond_t": True,
    }

    if is_causal:
        model_cls = QwenImageTransformerCausal2DModel
        pipe_cls = QwenImageEditPlusCausalPipeline
    else:
        model_cls = QwenImageTransformer2DModelFixSpeed
        pipe_cls = QwenImageEditPlusPipelineForBench

    with init_empty_weights(include_buffers=False):
        model = model_cls.from_config(config)
        model.to(torch_dtype)
    model.to_empty(device=device)

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

    pipe = pipe_cls.from_pretrained(
        model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
    )
    pipe.to(device)

    for i in range(3):
        cost_list = []
        for repeat in range(3):
            image_list = [
                Image.new("RGB", (1024, 1024), color="white") for _ in range(i + 1)
            ]
            input_args = {
                "prompt": "For test inference speed, a puppy is running",
                "generator": torch.Generator(device=device).manual_seed(0),
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": " ",
                "num_inference_steps": num_inference_steps,
                "image": image_list,
                "return_dit_cost": True,
            }
            cost = pipe(**input_args)
            cost_list.append(cost)
        mean_cost = sum(cost_list[1:]) / (repeat - 1)
        print(
            f"dit cost average using SDPA attention backend for {i + 1} reference image(s) is {mean_cost}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_causal", type=int, default=1)
    parser.add_argument(
        "--model_name", type=str, default="lightx2v/Qwen-Image-Edit-Causal"
    )  # model_name does not affect the Dit Cost, it's determined by is_causal
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1)
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        is_causal=bool(args.is_causal),
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
    )
