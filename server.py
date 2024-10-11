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
from utils import base64_to_img, img_to_base64 , can_expand
import time
from safetensors.torch import load_file
import requests

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
    order_id : str
    image_url : str
    width : int
    height : int
    overlap_width : int
    num_inference_steps: int
    resize_option : str
    prompt_input: str = None
    margin_x : int = None
    margin_y : int = None

@app.post("/generate-image/")
async def generate_image(payload: ImagePayload):
    try:
        source = Image.open(requests.get(payload.image_url, stream= True).raw).convert("RGB")
        target_size = (payload.width, payload.height)
        overlap = payload.overlap_width
        num_inference_steps = payload.num_inference_steps
        margin_x = payload.margin_x
        margin_y = payload.margin_y
        resize_size = max(source.width, source.height)

        logger.info(f"Recieved Data for order id {payload.order_id}")
    except:
        status_code = 400
        logger.error(traceback.format_exc())
        logger.info(f"[AI-Image-Extender] : Error in generating image for order id {payload.order_id}")
        response_ =  {'message': traceback.format_exc(), 'status_code':status_code}
        return JSONResponse(status_code=status_code, content=jsonable_encoder(response_))
    
    aspect_ratio = source.height / source.width
    new_width = resize_size
    new_height = int(resize_size * aspect_ratio)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    mask_draw.rectangle([
        (margin_x + overlap, margin_y + overlap),
        (margin_x + source.width - overlap, margin_y + source.height - overlap)
    ], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    
    final_prompt = f"high quality, 4k" if payload.prompt_input is None else payload.prompt_input
    t1 = time.time()
    try:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(final_prompt, "cuda", True)

        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=num_inference_steps
        ):
            cnet_image, image = cnet_image, image

        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), mask)
        # logger.info(f"mask size : {mask.size}  --- Image size : {result_image.size}")
        
        output_b64 = img_to_base64(cnet_image)

        response_ = {'message': "success", 'status_code': 200, 'output_image' : output_b64}
        logger.info(f"[AI-Image-Extender] : Image generated for order id {payload.order_id} in {time.time() - t1} seconds")
        torch.cuda.empty_cache()
        return JSONResponse(status_code=200, content=jsonable_encoder(response_))
    except:
        
        status_code = 500
        logger.error(traceback.format_exc())
        logger.info(f"[AI-Image-Extender] : Error in generating image for order id {payload.order_id}")
        print(f"[AI-Image-Extender] : Error in generating image for order id {payload.order_id}")
        response_ =  {'message': traceback.format_exc(), 'status_code':status_code}
        return JSONResponse(status_code=status_code, content=jsonable_encoder(response_))

