import gradio as gr
# import spaces
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from src.controlnet_union import ControlNetModel_Union
from src.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw
import numpy as np

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
state_dict = load_state_dict(model_file)
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


def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

# @spaces.GPU(duration=24)
def infer(image, width, height, overlap_width, num_inference_steps, resize_option, custom_resize_size, prompt_input=None, alignment="Middle"):
    source = image
    target_size = (width, height)
    overlap = overlap_width

    # Upscale if source is smaller than target in both dimensions
    if source.width < target_size[0] and source.height < target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)

    if source.width > target_size[0] or source.height > target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)
    
    if resize_option == "Full":
        resize_size = max(source.width, source.height)
    elif resize_option == "1/2":
        resize_size = max(source.width, source.height) // 2
    elif resize_option == "1/3":
        resize_size = max(source.width, source.height) // 3
    elif resize_option == "1/4":
        resize_size = max(source.width, source.height) // 4
    else:  # Custom
        resize_size = custom_resize_size

    aspect_ratio = source.height / source.width
    new_width = resize_size
    new_height = int(resize_size * aspect_ratio)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    if not can_expand(source.width, source.height, target_size[0], target_size[1], alignment):
        alignment = "Middle"

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - source.width
        margin_y = (target_size[1] - source.height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - source.width) // 2
        margin_y = target_size[1] - source.height

    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Adjust mask generation based on alignment
    if alignment == "Middle":
        mask_draw.rectangle([
            (margin_x + overlap, margin_y + overlap),
            (margin_x + source.width - overlap, margin_y + source.height - overlap)
        ], fill=0)
    elif alignment == "Left":
        mask_draw.rectangle([
            (margin_x, margin_y),
            (margin_x + source.width - overlap, margin_y + source.height)
        ], fill=0)
    elif alignment == "Right":
        mask_draw.rectangle([
            (margin_x + overlap, margin_y),
            (margin_x + source.width, margin_y + source.height)
        ], fill=0)
    elif alignment == "Top":
        mask_draw.rectangle([
            (margin_x, margin_y),
            (margin_x + source.width, margin_y + source.height - overlap)
        ], fill=0)
    elif alignment == "Bottom":
        mask_draw.rectangle([
            (margin_x, margin_y + overlap),
            (margin_x + source.width, margin_y + source.height)
        ], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = f"{prompt_input} , high quality, 4k"

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
        yield cnet_image, image

    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)

    yield background, cnet_image

def clear_result():
    """Clears the result ImageSlider."""
    return gr.update(value=None)

def preload_presets(target_ratio, ui_width, ui_height):
    """Updates the width and height sliders based on the selected aspect ratio."""
    if target_ratio == "9:16":
        changed_width = 720
        changed_height = 1280
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "16:9":
        changed_width = 1280
        changed_height = 720
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "1:1":
        changed_width = 1024
        changed_height = 1024
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "Custom":
        return ui_width, ui_height, gr.update(open=True)

def select_the_right_preset(user_width, user_height):
    if user_width == 720 and user_height == 1280:
        return "9:16"
    elif user_width == 1280 and user_height == 720:
        return "16:9"
    elif user_width == 1024 and user_height == 1024:
        return "1:1"
    else:
        return "Custom"

def toggle_custom_resize_slider(resize_option):
    return gr.update(visible=(resize_option == "Custom"))

def update_history(new_image, history):
    """Updates the history gallery with the new image."""
    if history is None:
        history = []
    history.insert(0, new_image)
    return history

css = """
.gradio-container {
    width: 1200px !important;
}
"""

title = """<h1 align="center">Diffusers Image Outpaint</h1>

"""

with gr.Blocks(css=css) as demo:
    with gr.Column():
        gr.HTML(title)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="Input Image"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_input = gr.Textbox(label="Prompt (Optional)")
                    with gr.Column(scale=1):
                        run_button = gr.Button("Generate")

                with gr.Row():
                    target_ratio = gr.Radio(
                        label="Expected Ratio",
                        choices=["9:16", "16:9", "1:1", "Custom"],
                        value="9:16",
                        scale=2
                    )
                    
                    alignment_dropdown = gr.Dropdown(
                        choices=["Middle", "Left", "Right", "Top", "Bottom"],
                        value="Middle",
                        label="Alignment"
                    )

                with gr.Accordion(label="Advanced settings", open=False) as settings_panel:
                    with gr.Column():
                        with gr.Row():
                            width_slider = gr.Slider(
                                label="Width",
                                minimum=720,
                                maximum=1536,
                                step=8,
                                value=720,  # Set a default value
                            )
                            height_slider = gr.Slider(
                                label="Height",
                                minimum=720,
                                maximum=1536,
                                step=8,
                                value=1280,  # Set a default value
                            )
                        with gr.Row():
                            num_inference_steps = gr.Slider(label="Steps", minimum=4, maximum=12, step=1, value=8)
                            overlap_width = gr.Slider(
                                label="Mask overlap width",
                                minimum=1,
                                maximum=50,
                                value=42,
                                step=1
                            )
                        with gr.Row():
                            resize_option = gr.Radio(
                                label="Resize input image",
                                choices=["Full", "1/2", "1/3", "1/4", "Custom"],
                                value="Full"
                            )
                            custom_resize_size = gr.Slider(
                                label="Custom resize size",
                                minimum=64,
                                maximum=1024,
                                step=8,
                                value=512,
                                visible=False
                            )
                            
                gr.Examples(
                    examples=[
                        ["./examples/example_1.webp", 1280, 720, "Middle"],
                        ["./examples/example_2.jpg", 1440, 810, "Left"],
                        ["./examples/example_3.jpg", 1024, 1024, "Top"],
                        ["./examples/example_3.jpg", 1024, 1024, "Bottom"],
                    ],
                    inputs=[input_image, width_slider, height_slider, alignment_dropdown],
                )

            with gr.Column():
                result = ImageSlider(
                    interactive=False,
                    label="Generated Image",
                )
                use_as_input_button = gr.Button("Use as Input Image", visible=False)

                history_gallery = gr.Gallery(label="History", columns=6, object_fit="contain", interactive=False)

        

    def use_output_as_input(output_image):
        """Sets the generated output as the new input image."""
        return gr.update(value=output_image[1])

    use_as_input_button.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_image]
    )
    
    target_ratio.change(
        fn=preload_presets,
        inputs=[target_ratio, width_slider, height_slider],
        outputs=[width_slider, height_slider, settings_panel],
        queue=False
    )

    width_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    height_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    resize_option.change(
        fn=toggle_custom_resize_slider,
        inputs=[resize_option],
        outputs=[custom_resize_size],
        queue=False
    )
    
    run_button.click(  # Clear the result
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(  # Generate the new image
        fn=infer,
        inputs=[input_image, width_slider, height_slider, overlap_width, num_inference_steps,
                resize_option, custom_resize_size, prompt_input, alignment_dropdown],
        outputs=result,
    ).then(  # Update the history gallery
        fn=lambda x, history: update_history(x[1], history),
        inputs=[result, history_gallery],
        outputs=history_gallery,
    ).then(  # Show the "Use as Input Image" button
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

    prompt_input.submit(  # Clear the result
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(  # Generate the new image
        fn=infer,
        inputs=[input_image, width_slider, height_slider, overlap_width, num_inference_steps,
                resize_option, custom_resize_size, prompt_input, alignment_dropdown],
        outputs=result,
    ).then(  # Update the history gallery
        fn=lambda x, history: update_history(x[1], history),
        inputs=[result, history_gallery],
        outputs=history_gallery,
    ).then(  # Show the "Use as Input Image" button
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

demo.queue(max_size=12).launch(share=True)