from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch
import base64
import os
from os import makedirs
from io import BytesIO
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, TCDScheduler
from huggingface_hub import hf_hub_download
from src.controlnet_union import ControlNetModel_Union
from src.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
import logging
from logging.handlers import RotatingFileHandler
import traceback
from utils import base64_to_img, img_to_base64
import time
from safetensors.torch import load_file

logger = logging.getLogger('AI-Image-Extender')
logger.setLevel(logging.INFO)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
makedirs(os.path.join(BASE_PATH, 'logs'), exist_ok=True)
path = os.path.join(BASE_PATH, 'logs', 'ai-image-extender.log')
handler = RotatingFileHandler(path,
                              maxBytes=2 * 1024 * 1024,
                              backupCount=5)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("[AI-Image-Extender] : Server has started and logging is set up.")

app = FastAPI()

logger.info("[AI-Image-Extender] : Initializing pipeline...")
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
# state_dict = torch.load(model_file)
state_dict = load_file(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to("cuda")
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

logger.info("[AI-Image-Extender] : Pipeline has been initialized.")


class ImagePayload(BaseModel):
    order_id: str
    c_net_base64: str
    mask_base64: str
    num_inference_steps: int
    prompt_input: str = None

@app.post("/generate-image/")
async def generate_image(payload: ImagePayload):
    cnet_image = base64_to_img(payload.c_net_base64)
    # cnet_image.save("cnet_image.png")
    mask = base64_to_img(payload.mask_base64)
    # mask.save("mask.png")
    # num_inference_steps = 10 if payload.num_inference_steps is None else payload.num_inference_steps
    final_prompt = f"high quality, 4k" if payload.prompt_input is None else payload.prompt_input
    t1 = time.time()
    try:
        (
            prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(final_prompt, "cuda", True)

        for result_image in pipe(
         prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
            num_inference_steps=payload.num_inference_steps,
        ):
            cnet_image, result_image = cnet_image, result_image

        result_image = result_image.convert("RGBA")

        cnet_image.paste(result_image, (0, 0), mask)
        cnet_image.save("cnet_image.png")
        result_image.save("result_image.png")

        logger.info(f"mask size : {mask.size}  --- Image size : {result_image.size}")
        
        output_b64 = img_to_base64(result_image)

        response_ = {'message': "success", 'status_code': 200, 'output_image' : output_b64}
        logger.info(f"[AI-Image-Extender] : Image generated for order id {payload.order_id} in {time.time() - t1} seconds")
        return JSONResponse(status_code=200, content=jsonable_encoder(response_))
    except:
        
        status_code = 500
        logger.error(traceback.format_exc())
        logger.info(f"[AI-Image-Extender] : Error in generating image for order id {payload.order_id}")
        print(f"[AI-Image-Extender] : Error in generating image for order id {payload.order_id}")
        response_ =  {'message': traceback.format_exc(), 'status_code':status_code}
        return JSONResponse(status_code=status_code, content=jsonable_encoder(response_))

