import requests

# url = "http://localhost:8000/generate-image"
url = "http://164.52.212.87:8000/generate-image"

payload = {
    "order_id": "1234567890",
    # "image_url": "https://ai-image-editor-wasabi-bucket.apyhi.com/uncrop_assets/original_images/image%2010.webp",
    "image_url": "https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/11bd44a0-d2ee-479d-889d-b5a048b3a157.jpeg",
    "width": 1536,
    "height": 1024,
    "scale":  0.5236363636363637,
    "num_inference_steps": 6,
    "margin_x": 100,
    "margin_y": 100,
    "user_type": "PAID"
}
response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())

# image_base64_list = response.json()["output_image_url"]
# import base64
# i=2
# for image_base64 in image_base64_list:
#     image = base64.b64decode(image_base64)
#     with open(f"output_{i}.png", "wb") as f:
#         f.write(image)
#     i += 1

