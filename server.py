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
import base64
from io import BytesIO
import time
from safetensors.torch import load_file
import requests
from starlette.requests import Request
from pprint import pprint
import uuid
from datetime import datetime


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
)
class ImageExtenderDeployment:
    def __init__(self):
        self.logger = logging.getLogger('AI-Image-Extender')
        self.logger.setLevel(logging.INFO)
        
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(BASE_PATH, 'logs'), exist_ok=True)
        log_path = os.path.join(BASE_PATH, 'logs', 'ai-image-extender.log')
        
        handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("[AI-Image-Extender] : Initializing pipeline...")
        self._initialize_models()
        self.logger.info("[AI-Image-Extender] : Pipeline has been initialized.")

    def img_to_base64(self,img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def get_output_urls(self,image):
        static_url = "https://static-cb2.phot.ai/extender/"
        date_folder = datetime.now().strftime("%Y-%m-%d")
        output_dir = "/storage/images_log"
        os.makedirs(output_dir, exist_ok=True)
        save_dir = os.path.join(output_dir, date_folder)
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{str(uuid.uuid4())}.webp"
        image.save(os.path.join(save_dir, file_name))
        return os.path.join(static_url, date_folder, file_name)


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
        self.pipe.enable_vae_slicing()
        

    def add_watermark(self,order_id, image):
    
        try:
            image = image.convert("RGBA")
            watermark = Image.open(os.path.join("assets", "watermark.png")).convert("RGBA")
            main_width, main_height = image.size
            watermark_height = int(main_height * 0.07)  # 7% of the main image's height
            watermark = watermark.resize((int(watermark.width * (watermark_height / watermark.height)), watermark_height))
            watermark_width = watermark.width
            x_position = (main_width - watermark_width) // 2  # Centered on x-axis
            y_position = main_height - watermark_height - int(main_height * 0.02)
            image.paste(watermark, (x_position, y_position), watermark)
            self.logger.info(f"Watermark added for order id {order_id}")
            return image.convert("RGB")
        except:
            self.logger.info(f"Error in adding watermark for order id {order_id}")
            self.logger.info(traceback.format_exc())
            return image

    def prepare_image(self,image,width,height,margin_x,margin_y,scale):
        new_input_width = int(image.width*scale)
        new_input_height = int(image.height*scale)
        input_image = image.resize((new_input_width,new_input_height), Image.LANCZOS)
        overlap_x = 42
        overlap_y = 42
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
                self.logger.error(f"Wrong url path {request_path} or method {request_method}")
                return {
                "message": "Wrong url path",
                "status_code": 200,
            }
            payload = await request.json()

            try :   
                image_url = payload.get("image_url")
                width = payload.get("width")
                height = payload.get("height")
                num_inference_steps = payload.get("num_inference_steps")
                margin_x = payload.get("margin_x")
                margin_y = payload.get("margin_y")
                scale = payload.get("scale")
                order_id = payload.get("order_id")
                user_type = payload.get("user_type") 

                source = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
                
                self.logger.info(f"Received Data for order id {order_id}")
                self.logger.info(f"Source image size: {source.size}")

            except Exception as e:
                self.logger.error(traceback.format_exc())
                self.logger.info(f"Error in generating image for order id {order_id}")
                torch.cuda.empty_cache()
                return {
                    "message": str(e),
                    "status_code": 422,
                    "error": traceback.format_exc(),
                    "output_doc": None,
                    "model_inference_time": None
                }
            # Resize if necessary
            # if source.size[0] > 2048 or source.size[1] > 2048:
            #     max_length = 2048
            #     aspect_ratio = source.size[0] / source.size[1]
            #     if aspect_ratio > 1:    
            #         new_width = max_length
            #         new_height = int(max_length / aspect_ratio)
            #     else:
            #         new_width = int(max_length * aspect_ratio)
            #         new_height = max_length    
            #     source = source.resize((new_width, new_height), Image.LANCZOS)
            #     logger.info(f"Resized image to {source.size}")
            try :
                t1 = time.time()
                background, mask = self.prepare_image(source, width, height, margin_x, margin_y, scale)
                cnet_image = background.copy()
                cnet_image.paste(0, (0, 0), mask)
                t2 = time.time()
                final_prompt = f"high quality, 4k" 
            
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

                image = image.convert("RGBA")
                image = image.resize(mask.size)
                cnet_image.paste(image, (0, 0), mask)
            
                # output_b64 = img_to_base64(cnet_image)
                
                image_without_watermark = cnet_image
                t3 = time.time()
                model_inference_time = t3 - t2
                self.logger.info(f"Model inference time : {model_inference_time}")
                image_without_watermark_url = self.get_output_urls(image_without_watermark)
                if user_type == "FREE":
                    image_with_watermark = self.add_watermark(order_id, image_without_watermark)
                    image_with_watermark_url = self.get_output_urls(image_with_watermark)
                self.logger.info(f"Total Processing time for order id {order_id} in {time.time() - t1} seconds")
                
                output_doc = {"0" : {"without_watermark" : image_without_watermark_url , "with_watermark" : image_with_watermark_url if user_type == "FREE" else None}}

                
                torch.cuda.empty_cache()
                return {
                "message": "success",
                "status_code": 200,
                "output_doc": output_doc,
                "model_inference_time": model_inference_time
                }
            except Exception as e:
                self.logger.error(traceback.format_exc())
                self.logger.info(f"Model error for order id {order_id}")
                torch.cuda.empty_cache()
                return {
                    "message": str(e),
                    "status_code": 500,
                    "error": traceback.format_exc(),
                    "output_doc": None,
                    "model_inference_time": None
                }
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.info(f"Error in generating image for order id {order_id}")
            torch.cuda.empty_cache()
            return {
                "message": str(e),
                "status_code": 500,
                "error": traceback.format_exc(),
                "output_doc": None,
                "model_inference_time": None
            }

app = ray.init(namespace="image_extender", ignore_reinit_error=True,
               include_dashboard=True, dashboard_host='0.0.0.0')
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
image_extender=ImageExtenderDeployment.bind()