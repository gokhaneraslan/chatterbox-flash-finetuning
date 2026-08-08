# Chatterbox-Flash Fine-Tuning (Turkish & Multi-Language) 🚀

This repository provides an end-to-end training, fine-tuning, and inference framework for **Resemble AI's Chatterbox-Flash** zero-shot TTS model.

Chatterbox-Flash replaces standard autoregressive TTS decoders with a **parallel block-diffusion decoder**, enabling 9x–13x real-time speech generation. This repository implements **Smart Placeholder Mapping** and **LoRA Fine-Tuning** to adapt Chatterbox-Flash to non-English languages (e.g., Turkish) without architectural breaking changes or tensor shape mismatches.

---

## 🌟 Key Features

- **Smart Placeholder Vocab Adaptation**: Maps target language characters (`ğ, Ğ, ı, İ, ş, Ş`) directly to internal `[PLACEHOLDER]` tokens in `tokenizer.json` without resizing tensor shapes.
- **LoRA & Full Fine-Tuning**: Efficient PEFT training using `modules_to_save=["text_emb"]` to ensure new character embeddings are updated.
- **Fast Offline Feature Caching**: Pre-extracts S3Gen audio tokens and Voice Encoder embeddings into offline `.pt` tensors to speed up training epochs by 20x.
- **Non-Destructive Silero VAD**: Trims lead/trail silence using 16kHz VAD analysis while preserving full 24kHz audio fidelity.
- **Dual Dataset Support**: Parses both **LJSpeech CSV** (`metadata.csv`) and **JSON / JSONL** formats.
- **Real-Time Validation Callback**: Generates multi-sentence audio samples at regular step intervals during training.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gokhaneraslan/chatterbox-flash-finetuning.git
cd chatterbox-flash-finetuning

pip install -r requirements.txt

pip install --no-deps chatterbox-tts==0.1.7
```

### 2. Download Pretrained Models & Setup Tokenizer

Run the automated setup script to download base model checkpoints to `src/models/` and configure `tokenizer.json`:

```bash
python setup.py
```

---

## 📊 Dataset Preparation

Organize your dataset in either **LJSpeech CSV** or **JSON** format inside `data/raw/`:

### Option A: LJSpeech Format (`data/raw/metadata.csv`)
```text
audio_001|Ağaçların altındaki soğuk su kaynağından içtik.|Ağaçların altındaki soğuk su kaynağından içtik.
audio_002|Yapay zeka modelleri gün geçtikçe gelişiyor.|Yapay zeka modelleri gün geçtikçe gelişiyor.
```

### Option B: JSON Format (`data/raw/metadata.json`)
```json
[
  {"id": "audio_001", "text": "Ağaçların altındaki soğuk su kaynağından içtik."},
  {"id": "audio_002", "text": "Yapay zeka modelleri gün geçtikçe gelişiyor."}
]
```

Place corresponding audio files (`.wav` or `.mp3`) into `data/raw/wavs/`.

---

## 🏋️ Training Pipeline

### Step 1: Preprocess Dataset (Feature Extraction)

Extract offline features into `data/processed/`:

```bash
python src/preprocess.py \
    --metadata_path data/raw/metadata.csv \
    --wav_dir data/raw/wavs \
    --output_dir data/processed
```

### Step 2: Run Training

Start fine-tuning using LoRA (default) or YAML configuration:

```bash
python train.py
```

Generated sample audio files will be saved periodically in `inference_samples/step_N/`.

---

## 🔊 Inference & LoRA Export

### Synthesize Speech from Trained Checkpoint

```bash
python infer.py \
    --audio_prompt reference_wavs/test.wav \
    --lora_dir checkpoints/chatterbox_flash_tr_lora \
    --text "Merhaba, Chatterbox-Flash modeliyle Türkçe ses sentezliyoruz."
```

### Merge LoRA Weights into Standalone Model

Merge LoRA adapters into a single `t3_flash.safetensors` model for deployment:

```bash
python merge_lora.py \
    --base_model_dir src/models \
    --lora_dir checkpoints/chatterbox_flash_tr_lora \
    --output_dir merged_model
```

---
