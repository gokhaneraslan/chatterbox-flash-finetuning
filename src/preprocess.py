import os
import sys
import json
import csv
import logging
import argparse
import torch
import torchaudio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src._chatterbox_flash import ChatterboxFlashTTS
from src.config import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


cfg = TrainConfig()

def load_silero_vad():

    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        get_speech_timestamps = utils[0]
        logger.info("Silero VAD model successfully loaded.")
        return vad_model, get_speech_timestamps
    except Exception as e:
        logger.warning(f"Failed to load Silero VAD, skipping VAD trimming. Error: {e}")
        return None, None


def apply_silero_vad(wav_24k: torch.Tensor, sr: int, vad_model, get_speech_timestamps) -> torch.Tensor:

    if vad_model is None or get_speech_timestamps is None:
        return wav_24k

    resampler_16k = torchaudio.transforms.Resample(sr, cfg.vad_sample_rate)
    wav_16k = resampler_16k(wav_24k).squeeze(0)


    timestamps = get_speech_timestamps(wav_16k, vad_model, sampling_rate=cfg.vad_sample_rate)
    
    if not timestamps:
        return wav_24k

    start_sec = timestamps[0]['start'] / cfg.vad_sample_rate
    end_sec = timestamps[-1]['end'] / cfg.vad_sample_rate

    start_sample_24k = int(start_sec * cfg.s3_sample_rate)
    end_sample_24k = int(end_sec * cfg.s3_sample_rate)

    trimmed_wav_24k = wav_24k[:, start_sample_24k:end_sample_24k]
    
    if trimmed_wav_24k.shape[1] < int(0.5 * cfg.s3_sample_rate):
        return wav_24k

    return trimmed_wav_24k


def load_metadata(metadata_path: str, wav_dir: str):

    items = []
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    if metadata_path.endswith('.json'):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                file_id = entry.get("id")
                text = entry.get("text")
                if file_id and text:
                    items.append({"id": file_id, "text": text})

    elif metadata_path.endswith('.jsonl'):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    file_id = entry.get("id") or os.path.splitext(os.path.basename(entry.get("audio_path", "")))[0]
                    text = entry.get("text") or entry.get("normalized_text", "")
                    if file_id and text:
                        items.append({"id": file_id, "text": text})

    elif metadata_path.endswith('.csv') or metadata_path.endswith('.txt'):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = '|' if '|' in sample else ','
            reader = csv.reader(f, delimiter=delimiter)
            
            for row in reader:
                if len(row) >= 2:
                    file_id = row[0].strip()
                    text = row[2].strip() if len(row) >= 3 else row[1].strip()
                    items.append({"id": file_id, "text": text})
    
    return items


def preprocess(
    metadata_path: str = cfg.metadata_path,
    wav_dir: str = cfg.wav_dir,
    output_dir: str = cfg.preprocessed_dir,
    model_dir: str = cfg.model_dir,
    prompt_duration: float = cfg.prompt_duration,
    use_vad: bool = cfg.use_vad
):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
        logger.info(f"Loading Chatterbox-Flash model from local directory: '{model_dir}'...")
        tts_engine = ChatterboxFlashTTS.from_local(model_dir, device=device)
    else:
        logger.warning(f"Local model folder '{model_dir}' not found or empty. Falling back to Hugging Face Hub...")
        tts_engine = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device=device)

    vad_model, get_speech_timestamps = load_silero_vad() if use_vad else (None, None)

    metadata = load_metadata(metadata_path, wav_dir)
    logger.info(f"Total items found in metadata: {len(metadata)}")

    resampler_24k_to_16k = torchaudio.transforms.Resample(cfg.s3_sample_rate, cfg.s3_tokenizer_sample_rate).to(device)

    success_count = 0
    SPEECH_STOP_ID = cfg.speech_stop_id

    for item in tqdm(metadata, desc="Preprocessing"):
        try:
            file_id = item["id"]
            raw_text = item["text"]

            wav_path = os.path.join(wav_dir, f"{file_id}.wav")
            if not os.path.exists(wav_path):
                wav_path = os.path.join(wav_dir, f"{file_id}.mp3")
                if not os.path.exists(wav_path):
                    logger.warning(f"Audio file not found for ID: {file_id}, skipping.")
                    continue

            wav, sr = torchaudio.load(wav_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != cfg.s3_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, cfg.s3_sample_rate)
                wav = resampler(wav)

            if use_vad and vad_model is not None:
                wav = apply_silero_vad(wav, cfg.s3_sample_rate, vad_model, get_speech_timestamps)

            wav_24k_device = wav.to(device)

            wav_16k_device = resampler_24k_to_16k(wav_24k_device)

            with torch.no_grad():

                wav_np_24k = wav_24k_device.cpu().squeeze().numpy()
                spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np_24k], sample_rate=cfg.s3_sample_rate)
                speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()

                s_tokens, _ = tts_engine.s3gen.tokenizer(wav_16k_device.unsqueeze(0))
                raw_speech_tokens = torch.atleast_1d(s_tokens.squeeze().cpu())
                
                stop_speech_tensor = torch.tensor([SPEECH_STOP_ID], dtype=raw_speech_tokens.dtype)
                speech_tokens = torch.cat([raw_speech_tokens, stop_speech_tensor], dim=0)

                prompt_samples_16k = int(prompt_duration * cfg.s3_tokenizer_sample_rate)
                if wav_16k_device.shape[1] < prompt_samples_16k:
                    prompt_wav_16k = torch.nn.functional.pad(wav_16k_device, (0, prompt_samples_16k - wav_16k_device.shape[1]))
                else:
                    prompt_wav_16k = wav_16k_device[:, :prompt_samples_16k]

                p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav_16k.unsqueeze(0))
                prompt_tokens = torch.atleast_1d(p_tokens.squeeze().cpu())

                text_tokens = torch.atleast_1d(tts_engine.tokenizer.text_to_tokens(raw_text).squeeze(0).cpu())

            save_path = os.path.join(output_dir, f"{file_id}.pt")
            torch.save({
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
                "text_tokens": text_tokens,
                "raw_text": raw_text
            }, save_path)

            success_count += 1

        except Exception as e:
            logger.error(f"Error processing item '{item.get('id', 'unknown')}': {e}")
            continue

    logger.info(f"Preprocessing completed! Successfully processed {success_count}/{len(metadata)} items. Saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatterbox-Flash Feature Extraction Preprocessing Script")
    parser.add_argument("--metadata_path", type=str, default=cfg.metadata_path, help="Path to metadata.csv or JSON/JSONL file")
    parser.add_argument("--wav_dir", type=str, default=cfg.wav_dir, help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default=cfg.preprocessed_dir, help="Output directory for processed .pt files")
    parser.add_argument("--model_dir", type=str, default=cfg.model_dir, help="Directory containing local model checkpoints")
    parser.add_argument("--no_vad", action="store_true", help="Disable Silero VAD silence trimming")

    args = parser.parse_args()

    preprocess(
        metadata_path=args.metadata_path,
        wav_dir=args.wav_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        use_vad=not args.no_vad
    )