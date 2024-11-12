import requests

url = "http://localhost:8014/generate-image"

payload = {
    "order_id": "rishabh",
    "image_url": "https://ai-image-editor-wasabi-bucket.apyhi.com/uncrop_assets/original_images/image%2010.webp",
    "width": 1024,
    "height": 1024,
    "overlap_width": 100,
    "prompt_input": "",
    "num_inference_steps": 10,
    "resize_option": "idk",
    "margin_x": 10,
    "margin_y": 10,
}

response = requests.post(url, json=payload)
print(response.status_code)