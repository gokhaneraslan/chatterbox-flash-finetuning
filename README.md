# Chatterbox-Flash: Ultra-Fast Block-Diffusion Fine-Tuning & LoRA Kit 🎙️⚡

A modular, production-ready fine-tuning and inference infrastructure designed specifically for **Resemble AI's Chatterbox-Flash** model.

This framework enables lightning-fast adaptation of Chatterbox-Flash to new languages (such as Turkish) and custom voices using **LoRA (Low-Rank Adaptation)** or **Full Fine-Tuning**, coupled with **Smart Placeholder Vocabulary Adaptation**.

> ### 💡 Why Chatterbox-Flash? (Speed & Architectural Advantages)
> Unlike standard autoregressive TTS models that predict speech tokens sequentially 1-by-1 (leading to high latency and slow inference), **Chatterbox-Flash** utilizes a **Parallel Block-Diffusion Architecture**.
> 
> * **10x–20x Faster Generation:** Predicts speech tokens in parallel contiguous chunks (blocks of $D=16$ tokens) using self-calibrated Pointwise Mutual Information (PMI) scoring.
> * **Ultra-Low Latency / RTF:** Powered by **FlashInfer paged KV cache** and **CUDA graph capture**, achieving sub-second real-time factors (RTF) suitable for real-time conversational AI.
> * **High Acoustic Fidelity:** Retains the natural prosody and voice cloning fidelity of the original Chatterbox while completely eliminating autoregressive bottlenecks.

---

## 🧠 Training Strategies: LoRA vs. Full Fine-Tune

This repository supports both parameter-efficient adaptation and full-checkpoint training, controllable via `use_lora` in `src/config.py`.

### 1. LoRA Mode (`use_lora = True`) — RECOMMENDED
* **What is it?** LoRA (Low-Rank Adaptation) freezes the 520M LLaMA backbone and only trains low-rank adapter matrices alongside target language embeddings (`text_emb` and `speech_emb`).
* **Best for:** Small to medium datasets (**10 hours or less**), single-speaker voice cloning, or rapid language adaptation.
* **Benefits:** Prevents catastrophic forgetting, reduces GPU VRAM consumption by ~60%, speeds up training, and prevents overfitting on small datasets.
* **Output:** Generates a lightweight adapter folder (~350 MB) containing adapter weights and extended embeddings.

### 2. Full Fine-Tune (`use_lora = False`)
* **What is it?** Unfreezes and updates all 520M parameters of the T3 Flash LLaMA backbone.
* **Best for:** Massive multi-speaker datasets (**strictly larger than 50–100 hours**) where you want to completely re-align the model's fundamental acoustic representations.
* **Benefits:** Provides maximum flexibility for large-scale language pre-training.

---

## 🔤 Tokenizer & Smart Placeholder Vocabulary Adaptation

Chatterbox-Flash relies on a grapheme-based (character-level) tokenizer. To fine-tune the model on a new language with unique characters (e.g., Turkish: `ğ, ı, ş, ç, ö, ü, Ğ, İ, Ş`), we employ a **Placeholder Mapping Strategy**:

1. **Placeholder Allocation:** The base `tokenizer.json` downloaded from Hugging Face contains unassigned placeholder slots (e.g., `<unused_0>`, `<unused_1>`).
2. **Current Implementation (Manual Mapping):** We download `tokenizer.json` and manually assign target language special characters directly to these unused placeholder slots in the JSON file.
3. **Future Roadmap:** An automated vocabulary extension script (`extend_vocab.py`) will be integrated in an upcoming update to automatically discover and map missing graphemes from metadata text.
4. **Embedding Adaptation:** In `src/config.py`, setting `lora_modules_to_save = ["text_emb", "speech_emb"]` forces PEFT to keep the character embedding layers trainable. During fine-tuning, the model learns the exact acoustic/phonetic representations for these newly mapped placeholder tokens.

---

## 🛠️ Attention Masking: Training ↔ Inference Parity

Standard TTS implementations suffer from **padding leakage** when `batch_size > 1`: valid tokens erroneously attend to `PAD` tokens in the batch, causing acoustic degradation and stuttering. We also found that training needs to see the *same* attention pattern the real block-diffusion engine uses at inference time — otherwise the model learns something subtly different from what it will actually do at generation time.

We patched `T3.forward()` with a **4D Block-Causal Attention Mask** (`create_t3_block_causal_attention_mask`), reverse-engineered directly from the real inference engine: each speech block can see itself and every block before it, but never a future block — matching exactly how the KV cache is built during generation. All `PAD` positions (text and speech) are fully masked out, guaranteeing zero padding leakage.

A second, separate leak lived in the **Perceiver conditioning-prompt resampler**, which ignored padding entirely when batching reference prompts of different lengths. This is now masked too, via the same length-aware mechanism.

---


## 🚀 Installation

### 1. System Dependencies & FFmpeg
Make sure FFmpeg is installed on your system:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# MacOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 2. Python Environment
Clone the repository and install requirements:

```bash
git clone https://github.com/gokhaneraslan/chatterbox-flash-finetuning.git
cd chatterbox-flash-finetuning

pip install -r requirements.txt

python setup.py
```

---

## 📊 Dataset Preparation

### Option A: Using TTS Dataset Generator (Recommended)
We strongly recommend using the [TTS Dataset Generator](https://github.com/gokhaneraslan/tts-dataset-generator) tool to automatically process long audio or video files into LJSpeech format.

```bash
# Clone dataset generator
git clone https://github.com/gokhaneraslan/tts-dataset-generator.git
cd tts-dataset-generator
pip install -r requirements.txt

# Process video/audio file with Whisper AI
python main.py --file your_audio.mp4 --model large --language tr --ljspeech True
```
This tool automatically slices audio into 3–15 second chunks, transcribes text via Whisper AI, removes silence, and outputs an LJSpeech-formatted dataset ready for `data/raw/`.

### Option B: Manual Dataset Formatting
Place your dataset in `data/raw/`:
* `data/raw/metadata.csv` (LJSpeech format: `filename|raw_text|normalized_text` or `filename|text`)
* `data/raw/wavs/*.wav` (3–15 second mono/stereo audio clips)

---

## ⚡ Step-by-Step Training Workflow

### Step 1: Preprocessing (Offline Feature Extraction)
To maximize GPU utilization during training, run `src/preprocess.py`. This script extracts Voice Encoder speaker embeddings (24kHz), S3Tokenizer speech tokens (16kHz), prompt slices, and text tokens into offline `.pt` cache files:

```bash
python src/preprocess.py
```

### Step 2: Configure Parameters
Edit `src/config.py` to match your hardware and training goals:

```python
# In src/config.py
use_lora: bool = True               # True: Fast LoRA Training | False: Full Fine-Tuning
batch_size: int = 16                # Batch size per GPU
grad_accum: int = 2                 # Gradient accumulation steps
learning_rate: float = 1e-4         # 1e-4 for LoRA, 2e-5 for Full
num_epochs: int = 5                 # 3-5 epochs is optimal for large datasets
block_size: int = 16                # Block size D for parallel block diffusion
```

### Step 3: Start Fine-Tuning
Launch training using PyTorch or HuggingFace Trainer pipeline:

```bash
python train.py
```
During training, evaluation audio samples will be generated periodically in `inference_samples/` so you can listen to voice quality progression in real time.

---

## 🗣️ Inference & LoRA Merging

### 1. Generating Speech (Inference)
Run zero-shot TTS inference using your reference speaker audio prompt and custom text:

```bash
python src/infer.py \
  --audio_prompt /path/to/reference_speaker.wav \
  --text "Merhaba, bu Chatterbox-Flash Türkçe test sesidir." \
  --lora_dir checkpoints/checkpoint-2000 \
  --output_dir inference_output
```

### 2. Merging LoRA Weights into Base Model
Once satisfied with the trained LoRA adapter, merge it into the base model to create a standalone, single-file deployment checkpoint:

```bash
python src/merge_lora.py \
  --base_model_dir src/models \
  --lora_dir checkpoints/checkpoint-2000 \
  --output_dir merged_model
```
This generates `merged_model/t3_flash.safetensors` alongside required companion files (`ve.safetensors`, `s3gen.safetensors`, `tokenizer.json`), ready for production deployment without requiring PEFT at inference time.

---

## 🙏 Acknowledgments
* Based on **Chatterbox-Flash** by [Resemble AI](https://github.com/resemble-ai/chatterbox). Special thanks to Resemble AI for pioneering block-diffusion zero-shot speech synthesis.
* Dataset preparation tool: [tts-dataset-generator](https://github.com/gokhaneraslan/tts-dataset-generator).