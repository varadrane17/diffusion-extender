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



# NUM_TO_STRING_MAP = {
#     "16:9" : "WIDESCREEN",
#     "9:16" : "MOBILE_VERTICAL",
#     "3:2" : "LANDSCAPE",
#     "4:5" : "PORTRAIT",
#     "1:1" : "SQUARE",
#     "CUSTOM" : "CUSTOM",
#     "SQUARE" : "SQUARE",
#     "PORTRAIT" : "PORTRAIT",
#     "LANDSCAPE" : "LANDSCAPE",
#     "WIDESCREEN" : "WIDESCREEN",
#     "MOBILE_VERTICAL" : "MOBILE_VERTICAL",
#     "custom" : "custom"
# }
# ASPECT_RATIO_TO_VALUE_MAP = {
#     "WIDESCREEN": 16/9,
#     "MOBILE_VERTICAL": 9/16,
#     "LANDSCAPE": 3/2,
#     "PORTRAIT": 4/5,
#     "SQUARE": 1.0,
# }

# def check_overlap(input_image_size, aspect_ratio):
#     if ASPECT_RATIO_TO_VALUE_MAP[aspect_ratio] == input_image_size[0] / input_image_size[1]:
#         return True
#     return False

# def check_custom_overlap(input_image_size, canvas_w, canvas_h, center_x, center_y, scale):
#     input_aspect_ratio = input_image_size[0] / input_image_size[1]
#     canvas_aspect_ratio = canvas_w / canvas_h
#     if input_aspect_ratio == canvas_aspect_ratio and center_x == 0 and center_y == 0 and scale == 1:
#         return True
#     return False

# def process_aspect_ratio(input_image_size, aspect_ratio):

#     if aspect_ratio == "SQUARE":
#         max_dim = max(input_image_size) 
#         height = width = max_dim
#         margin_x = margin_y = (max_dim - input_image_size[0]) // 2
#         return height, width, margin_x, margin_y
    
#     elif aspect_ratio == "LANDSCAPE":
#         height = input_image_size[1]
#         width = int(height * 3 / 2)
#         margin_x = (width - input_image_size[0]) // 2
#         margin_y = 0

#     elif aspect_ratio == "WIDESCREEN":
#         height = input_image_size[1]
#         width = int(height * 16 / 9)
#         margin_x = (width - input_image_size[0]) // 2
#         margin_y = 0

#     elif aspect_ratio == "PORTRAIT":
#         width = input_image_size[0]
#         height = int(width * 5 / 4)
#         margin_x = 0
#         margin_y = (height - input_image_size[1]) // 2

#     elif aspect_ratio == "MOBILE_VERTICAL" :
#         width = input_image_size[0]
#         height = int(width * 16 / 9)
#         margin_x = 0
#         margin_y = (height - input_image_size[1]) // 2

#     else:
#         raise ValueError("Invalid aspect ratio")

#     return height, width, margin_x, margin_y

# def process_custom_aspect_ratio(input_image_size, canvas_w, canvas_h, center_x, center_y, scale):
#    # center_x, center_y are x y position of the image on the canvas with 0,0 being the center of the canvas
#    # scale is the scale of the image
#    # canvas_w, canvas_h are the width and height of the canvas
#    # input_image_size is the width and height of the input image

#     input_image_w, input_image_h = input_image_size
#     scaled_w = input_image_w * scale
#     scaled_h = input_image_h * scale
#     top_left_x = center_x - scaled_w / 2
#     top_left_y = center_y - scaled_h / 2
#     margin_x = (canvas_w / 2) + top_left_x
#     margin_y = (canvas_h / 2) + top_left_y
    
#     return canvas_h, canvas_w, margin_x, margin_y


class ImagePayload(BaseModel):
    order_id : str
    # image_url : str
    image : str
    mask : str
    c_net_image : str
    # width : int
    # height : int
    # overlap_width : int
    num_inference_steps: int
    # resize_option : str

@app.post("/generate-image/")
async def generate_image(payload: ImagePayload):
    try:
        cnet_image = base64_to_img(payload.c_net_image)
        mask = base64_to_img(payload.mask)
        image = base64_to_img(payload.image)
        num_inference_steps = payload.num_inference_steps
        


        logger.info(f"Recieved Data for order id {payload.order_id}")
        logger.info(f"Source image size : {cnet_image.size}")
    except:
        status_code = 400
        logger.error(traceback.format_exc())
        logger.info(f"[AI-Image-Extender] : Error in generating image for order id {payload.order_id}")
        response_ =  {'message': traceback.format_exc(), 'status_code':status_code}
        return JSONResponse(status_code=status_code, content=jsonable_encoder(response_))
    
    # if source.size[0] > 2048 or source.size[1] > 2048:
    #     max_length = 2048
    #     # max_dim = max(source.size)
    #     aspect_ratio = source.size[0] / source.size[1]   # aspect ratio is width / height = 2498 / 3747 = 0.6666666666666666
    #     # width, height = source.size

    #     if aspect_ratio > 1:    
    #         new_width = max_length
    #         new_height = int(max_length / aspect_ratio)
    #     else:
    #         new_width = int(max_length * aspect_ratio)  # new_width = int(2048 * 0.6666666666666666) = 1366
    #         new_height = max_length    
    #     source = source.resize((new_width, new_height), Image.LANCZOS)

    # logger.info(f"Resized image to {source.size}")

    # background = Image.new('RGB', target_size, (255, 255, 255))
    # background.paste(source, (margin_x, margin_y))

    # mask = Image.new('L', target_size, 255)
    # logger.info(f"Mask size : {mask.size}")
    # mask_draw = ImageDraw.Draw(mask)

    # mask_draw.rectangle([
    #     (margin_x + overlap, margin_y + overlap),
    #     (margin_x + source.width - overlap, margin_y + source.height - overlap)
    # ], fill=0)

    # cnet_image = background.copy()
    # cnet_image.paste(0, (0, 0), mask)
    # logger.info(f"Mask size : {mask.size}")
    # logger.info(f"Cnet image size : {cnet_image.size}")

    
    final_prompt = f"high quality, 4k"
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
        # print(mask.mode)
        mask = mask.convert("RGBA")
        # mask = mask.resize(image.size)
        image = image.resize(mask.size)
        cnet_image.paste(image, (0, 0), mask)
        logger.info(f"Newmask size : {mask.size}  --- Image size : {cnet_image.size}")
        
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

