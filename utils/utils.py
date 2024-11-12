import os 
from PIL import Image
import base64
from io import BytesIO
import io


def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def base64_to_img(img_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(img_str)))

def img_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
    