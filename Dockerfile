# Use a base image with Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

RUN apt-get update -y && \
	apt-get install -y aria2 libgl1 libglib2.0-0 wget gifsicle libimage-exiftool-perl

# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
