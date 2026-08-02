import os
import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # --- Paths ---
    model_dir: str = "models"                             # Directory containing downloaded pretrained models
    metadata_path: str = "data/raw/metadata.csv"         # Path to dataset metadata (CSV, JSON, or JSONL)
    wav_dir: str = "data/raw/wavs"                        # Directory containing raw audio files (.wav or .mp3)
    preprocessed_dir: str = "data/processed"              # Output directory for extracted .pt feature tensors
    output_dir: str = "checkpoints"                      # Output directory for trained model checkpoints/LoRA weights
    sample_output_dir: str = "inference_samples"          # Output directory for generated test audio during training

    # --- Audio & VAD Settings ---
    s3_sample_rate: int = 24000                           # Chatterbox S3Gen vocoder sample rate (24kHz)
    vad_sample_rate: int = 16000                          # Silero VAD sample rate (16kHz)
    prompt_duration: float = 3.0                          # Reference audio prompt duration in seconds
    use_vad: bool = True                                  # Enable non-destructive Silero VAD silence trimming

    # --- Tokenizer & Vocabulary Constants (No Magic Numbers!) ---
    text_vocab_size: int = 704                            # Default Chatterbox text tokenizer vocabulary size
    start_text_token: int = 255                           # [START] text token ID
    stop_text_token: int = 0                              # [STOP] text token ID
    speech_stop_id: int = 6562                            # [STOP_SPEECH] token ID for S3Tokenizer
    mask_token_id: int = 6561                             # [MASK] speech token ID for Block Diffusion
    pad_token_id: int = 0                                 # Padding token ID

    # --- Sequence Constraints ---
    max_text_len: int = 256                               # Maximum text token length
    max_speech_len: int = 850                             # Maximum speech token length

    # --- Block Diffusion & Masking Strategy ---
    block_size: int = 16                                  # Block size D for parallel block diffusion
    min_mask_ratio: float = 0.15                          # Minimum masking ratio during training
    max_mask_ratio: float = 1.0                           # Maximum masking ratio during training
    uncond_prob: float = 0.15                             # CFG (Classifier-Free Guidance) dropout probability

    # --- Training Hyperparameters ---
    use_lora: bool = True                                 # True: Train LoRA adapters | False: Full Fine-Tuning
    batch_size: int = 16                                  # Training batch size per GPU
    grad_accum: int = 1                                   # Gradient accumulation steps
    learning_rate: float = 1e-4                           # Learning rate (1e-4 for LoRA, 1e-5 for Full)
    weight_decay: float = 0.01                            # Weight decay for AdamW optimizer
    num_epochs: int = 10                                  # Total number of training epochs
    warmup_steps: int = 100                               # Linear warmup steps
    save_steps: int = 500                                 # Checkpoint save interval in steps
    save_total_limit: int = 5                             # Maximum number of checkpoints to retain
    dataloader_num_workers: int = 4                       # PyTorch DataLoader worker threads
    mixed_precision: str = "bf16"                         # Precision mode: 'bf16', 'fp16', or 'fp32'

    # --- LoRA Parameters ---
    lora_r: int = 128                                     # LoRA rank dimension
    lora_alpha: int = 256                                 # LoRA scaling alpha
    lora_dropout: float = 0.05                            # LoRA dropout probability
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    lora_modules_to_save: List[str] = field(
        default_factory=lambda: ["text_emb"] # CRITICAL: Saves & trains Turkish placeholder embeddings
    )

    # --- Multi-Sentence Evaluation / Audio Sample Callback ---
    eval_prompt_path: Optional[str] = "data/raw/reference.wav"
    eval_steps: int = 500   # Generate sample audio every N steps
    eval_sample_texts: List[str] = field(
        default_factory=lambda: [
            "Merhaba, bu Chatterbox-Flash Türkçe fine-tuning test sesidir.",
            "Ağaçların altındaki soğuk su kaynağından taze su içtik.",
            "Yapay zeka teknolojileri her geçen gün daha da gelişiyor, değil mi?",
            "Ses modelimiz şu an Türkçe dil kalıplarını başarıyla öğrenmektedir."
        ]
    )
    
    def save_yaml(self, yaml_path: str):
        """Saves configuration to a YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuration saved to YAML at: '{yaml_path}'")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainConfig":
        """Loads configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration YAML file not found: '{yaml_path}'")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.info(f"Configuration loaded from YAML at: '{yaml_path}'")
        return cls(**data)