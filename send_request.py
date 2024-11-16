import requests

url = "http://localhost:8000/generate-image"

payload = {
    "order_id": "1234567890",
    "image_url": "https://ai-image-editor-wasabi-bucket.apyhi.com/uncrop_assets/original_images/image%2010.webp",
    "width": 1024,
    "height": 1024,
    "scale":  1.066,
    "num_inference_steps": 10,
    "margin_x": 100,
    "margin_y": 100,
}
response = requests.post(url, json=payload)
print(response.status_code)