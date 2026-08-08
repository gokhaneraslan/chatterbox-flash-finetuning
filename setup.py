import os
import sys
import logging
import requests
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SetupModels")

MODELS_DIR = "src/models"
HF_REPO_ID = "ResembleAI/chatterbox-flash"

MODEL_FILES = {
    "s3gen.safetensors": f"https://huggingface.co/{HF_REPO_ID}/resolve/main/s3gen.safetensors?download=true",
    "t3_flash.safetensors": f"https://huggingface.co/{HF_REPO_ID}/resolve/main/t3_flash.safetensors?download=true",
    "ve.safetensors": f"https://huggingface.co/{HF_REPO_ID}/resolve/main/ve.safetensors?download=true"
}


def download_file(url: str, destination_path: str):

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    filename = os.path.basename(destination_path)
    with open(destination_path, "wb") as file, tqdm(
        desc=f"Downloading {filename}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            size = file.write(data)
            progress_bar.update(size)


def setup_pretrained_models():

    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("==================================================")
    logger.info("   Setting Up Chatterbox-Flash Pretrained Models  ")
    logger.info("==================================================")

    for filename, url in MODEL_FILES.items():
        file_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"File '{filename}' already exists in '{MODELS_DIR}/', skipping download.")
        else:
            logger.info(f"Downloading '{filename}'...")
            try:
                download_file(url, file_path)
                logger.info(f"Successfully downloaded '{filename}'.")
            except Exception as e:
                logger.error(f"Failed to download '{filename}' from URL: {url}. Error: {e}")
                sys.exit(1)

    logger.info("==================================================")
    logger.info("Setup complete! All models ready in 'models/' directory.")
    logger.info("==================================================")


if __name__ == "__main__":
    setup_pretrained_models()