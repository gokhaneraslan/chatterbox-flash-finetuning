import os
import sys
import shutil
import argparse
import logging
import torch
from peft import PeftModel
from safetensors.torch import save_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src._chatterbox_flash import ChatterboxFlashTTS
from src.config import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

cfg = TrainConfig()


def merge_lora_checkpoint(
    base_model_dir: str,
    lora_dir: str,
    output_dir: str
):

    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"LoRA directory not found: '{lora_dir}'")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(base_model_dir) and len(os.listdir(base_model_dir)) > 0:
        logger.info(f"Loading base Chatterbox-Flash model from local directory: '{base_model_dir}'...")
        tts = ChatterboxFlashTTS.from_local(base_model_dir, device=device)
    else:
        logger.info(f"Local base model directory '{base_model_dir}' not found. Downloading from HF Hub...")
        tts = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device=device)


    logger.info(f"Loading LoRA adapter from: '{lora_dir}'...")
    peft_t3 = PeftModel.from_pretrained(tts.t3, lora_dir)

    logger.info("Merging LoRA adapter weights directly into T3 Flash backbone...")
    merged_t3 = peft_t3.merge_and_unload()
    merged_t3.eval()

    output_t3_path = os.path.join(output_dir, "t3_flash.safetensors")
    logger.info(f"Saving merged T3 Flash model weights to: '{output_t3_path}'...")
    save_file(merged_t3.state_dict(), output_t3_path)

    companion_files = ["ve.safetensors", "s3gen.safetensors", "tokenizer.json"]
    for file_name in companion_files:
        src_file = os.path.join(base_model_dir, file_name)
        dst_file = os.path.join(output_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied companion file: '{file_name}' -> '{dst_file}'")
        else:
            logger.warning(f"Companion file '{file_name}' not found in '{base_model_dir}', skipped.")

    logger.info("==================================================================")
    logger.info(f"LoRA merge complete! Standalone deployment model ready at: '{output_dir}'")
    logger.info("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA Weights into Base Chatterbox-Flash Model")
    parser.add_argument("--base_model_dir", type=str, default=cfg.model_dir, help="Directory containing base pretrained model")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory containing trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="merged_model", help="Directory to save merged standalone model")

    args = parser.parse_args()

    merge_lora_checkpoint(
        base_model_dir=args.base_model_dir,
        lora_dir=args.lora_dir,
        output_dir=args.output_dir
    )