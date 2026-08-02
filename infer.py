import os
import sys
import argparse
import logging
import soundfile as sf
import torch
from pathlib import Path
from peft import PeftModel

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatterbox_flash import ChatterboxFlashTTS
from safetensors.torch import load_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_for_inference(
    model_dir: str = "models",
    lora_dir: str = None,
    full_checkpoint: str = None,
    device: str = "cuda"
) -> ChatterboxFlashTTS:
    """
    Loads base Chatterbox-Flash pipeline and applies trained LoRA weights or full checkpoint if provided.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. Load Base Chatterbox-Flash Pipeline
    if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
        logger.info(f"Loading base Chatterbox-Flash model from local directory: '{model_dir}'...")
        tts_engine = ChatterboxFlashTTS.from_pretrained(model_dir, device=device)
    else:
        logger.info(f"Local model directory '{model_dir}' not found. Downloading base model from HF Hub...")
        tts_engine = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device=device)

    # 2. Apply LoRA Adapter if provided
    if lora_dir and os.path.exists(lora_dir):
        logger.info(f"Applying fine-tuned LoRA adapter weights from: '{lora_dir}'...")
        tts_engine.t3 = PeftModel.from_pretrained(tts_engine.t3, lora_dir)
        tts_engine.t3.eval()

    # 3. Apply Full Fine-Tuned Checkpoint if provided
    elif full_checkpoint and os.path.exists(full_checkpoint):
        logger.info(f"Loading full fine-tuned safetensors checkpoint from: '{full_checkpoint}'...")
        state_dict = load_file(full_checkpoint)
        tts_engine.t3.load_state_dict(state_dict)
        tts_engine.t3.eval()

    else:
        logger.info("Using base model weights without fine-tuning adapters.")

    return tts_engine


def main():
    parser = argparse.ArgumentParser(description="Chatterbox-Flash Zero-Shot TTS Inference Script")
    parser.add_argument("--audio_prompt", type=str, required=True, help="Path to reference speaker .wav file")
    parser.add_argument("--text", type=str, nargs="+", default=None, help="One or more input sentences to synthesize")
    parser.add_argument("--text_file", type=str, default=None, help="Path to text file containing one sentence per line")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Directory to save generated .wav files")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory containing base pretrained model")
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to trained LoRA adapter directory")
    parser.add_argument("--full_checkpoint", type=str, default=None, help="Path to trained full safetensors checkpoint")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance (CFG) scale")

    args = parser.parse_args()

    # Collect Input Texts
    texts = []
    if args.text:
        texts.extend(args.text)
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            for line in f:
                line_clean = line.strip()
                if line_clean and not line_clean.startswith("#"):
                    texts.append(line_clean)

    if not texts:
        logger.error("No input texts provided! Pass --text or --text_file.")
        sys.exit(1)

    if not os.path.exists(args.audio_prompt):
        logger.error(f"Reference audio prompt file not found: '{args.audio_prompt}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model Pipeline
    tts = load_model_for_inference(
        model_dir=args.model_dir,
        lora_dir=args.lora_dir,
        full_checkpoint=args.full_checkpoint
    )

    logger.info(f"Preparing speaker conditioning from reference audio: '{args.audio_prompt}'...")
    conds = tts.prepare_conditionals(args.audio_prompt)
    conds_list = [conds] * len(texts)

    logger.info(f"Synthesizing {len(texts)} text utterance(s)...")
    
    with torch.inference_mode():
        wavs = tts.generate_batch(
            texts=texts,
            conds_list=conds_list,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            normalize_text=False
        )

    # Save Output Audio Files
    stem = Path(args.audio_prompt).stem
    for idx, (text, wav) in enumerate(zip(texts, wavs)):
        out_filename = f"{stem}_output_{idx:03d}.wav"
        out_path = os.path.join(args.output_dir, out_filename)
        wav_np = wav.detach().float().cpu().numpy().squeeze()
        sf.write(out_path, wav_np, tts.sr)
        logger.info(f"[{idx+1}/{len(texts)}] Generated: '{out_path}' | Text: '{text[:50]}...'")

    logger.info(f"Synthesis completed! All audio files saved to: '{args.output_dir}'")


if __name__ == "__main__":
    main()