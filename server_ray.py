import ray
from ray import serve
from typing import Dict, Any
import torch
import os
from os import makedirs
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, TCDScheduler
from huggingface_hub import hf_hub_download
from src.controlnet_union import ControlNetModel_Union
from src.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
import logging
from logging.handlers import RotatingFileHandler
import traceback
from utils import img_to_base64
import time
from safetensors.torch import load_file
import requests
from starlette.requests import Request
from pprint import pprint

logger = logging.getLogger('ray.serve')
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


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "max_replicas": 3,
    #     "target_num_ongoing_requests_per_replica": 10,

    # }
)
class ImageExtenderDeployment:
    def __init__(self):
        logger.info("[AI-Image-Extender] : Initializing pipeline...")
        self._initialize_models()
        logger.info("[AI-Image-Extender] : Pipeline has been initialized.")

    def _initialize_models(self):
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
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
        )
        model.to(device="cuda", dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
            variant="fp16",
        ).to("cuda")
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

    def prepare_image(self,image,height,width,margin_x,margin_y,scale):
        new_input_width = int(width*scale)
        new_input_height = int(height*scale)
        input_image = image.resize((new_input_width,new_input_height), Image.LANCZOS)
        overlap_x = int(new_input_width*margin_x)
        overlap_y = int(new_input_height*margin_y)

        margin_x = max(0, min(margin_x, new_input_width - new_input_width))
        margin_y = max(0, min(margin_y, new_input_height - new_input_height))

        background = Image.new('RGB', (width,height), (255, 255, 255))
        background.paste(input_image, (margin_x, margin_y))

        left_overlap = margin_x + overlap_x
        right_overlap = margin_x + new_input_width - overlap_x
        top_overlap = margin_y + overlap_y
        bottom_overlap = margin_y + new_input_height - overlap_y

        mask = Image.new('L', (width,height), 255)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([
            (left_overlap, top_overlap),
            (right_overlap, bottom_overlap)
        ], fill=0)

        return background, mask
        

    async def __call__(self, request: Request) -> Dict[str, Any]:
        try:
            request_path = request.url.path
            request_method = request.method
            if request_method != "POST" or request_path != "/generate-image":
                logger.error(f"Wrong url path {request_path} or method {request_method}")
                return {
                "message": "Wrong url path",
                "status_code": 200,
            }
            payload = await request.json()
            pprint(payload)
            # Load and process image
            image_url = payload.get("image_url")
            width = payload.get("width")
            height = payload.get("height")
            overlap_width = payload.get("overlap_width")
            num_inference_steps = payload.get("num_inference_steps")
            margin_x = payload.get("margin_x")
            margin_y = payload.get("margin_y")
            prompt_input = payload.get("prompt_input")
            order_id = payload.get("order_id")
            source = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            target_size = (width, height)
            
            logger.info(f"Received Data for order id {order_id}")
            logger.info(f"Source image size: {source.size}")

            # Resize if necessary
            if source.size[0] > 2048 or source.size[1] > 2048:
                max_length = 2048
                aspect_ratio = source.size[0] / source.size[1]
                if aspect_ratio > 1:    
                    new_width = max_length
                    new_height = int(max_length / aspect_ratio)
                else:
                    new_width = int(max_length * aspect_ratio)
                    new_height = max_length    
                source = source.resize((new_width, new_height), Image.LANCZOS)
                logger.info(f"Resized image to {source.size}")

            # Create background and mask
            background = Image.new('RGB', target_size, (255, 255, 255))
            background.paste(source, (margin_x, margin_y))

            mask = Image.new('L', target_size, 255)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle([
                (margin_x + overlap_width, 
                 margin_y + overlap_width),
                (margin_x + source.width - overlap_width, 
                 margin_y + source.height - overlap_width)
            ], fill=0)

            cnet_image = background.copy()
            cnet_image.paste(0, (0, 0), mask)

            # Generate image
            t1 = time.time()
            final_prompt = f"high quality, 4k" if prompt_input is None else prompt_input
            
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(final_prompt, "cuda", True)

            for image in self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                image=cnet_image,
                num_inference_steps=num_inference_steps
            ):
                cnet_image, image = cnet_image, image

            # Post-process image
            image = image.convert("RGBA")
            image = image.resize(mask.size)
            cnet_image.paste(image, (0, 0), mask)
            
            output_b64 = img_to_base64(cnet_image)
            
            logger.info(f"Image generated for order id {order_id} in {time.time() - t1} seconds")
            torch.cuda.empty_cache()
            
            return {
                "message": "success",
                "status_code": 200,
                "output_image": output_b64
            }

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.info(f"Error in generating image for order id {order_id}")
            torch.cuda.empty_cache()
            return {
                "message": str(e),
                "status_code": 500,
                "error": traceback.format_exc()
            }

app = ray.init(namespace="image_extender", ignore_reinit_error=True,
               include_dashboard=True, dashboard_host='0.0.0.0')
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8014})
image_extender=ImageExtenderDeployment.bind()