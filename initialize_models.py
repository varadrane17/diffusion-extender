import torch
import os
from os import makedirs
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, TCDScheduler
from huggingface_hub import hf_hub_download
from src.controlnet_union import ControlNetModel_Union
from src.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from safetensors.torch import load_file

def create_pipe():  
    config_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
    )
    config = ControlNetModel_Union.load_config(config_file)
    controlnet_model = ControlNetModel_Union.from_config(config)
    model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
    )
    state_dict = load_file(model_file)
    model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0")
    model.to(device="cuda", dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=torch.float16,
        vae=vae,
        controlnet=model,
        variant="fp16",
    )
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    return pipe

pipe = create_pipe()