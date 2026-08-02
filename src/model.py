import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from peft import LoraConfig, get_peft_model
from chatterbox_flash import ChatterboxFlashTTS
from src.config import TrainConfig

logger = logging.getLogger(__name__)


class ChatterboxFlashForTraining(nn.Module):
    """
    Training wrapper for Chatterbox-Flash T3 Block-Diffusion model.
    Encapsulates model initialization, LoRA/PEFT adaptation, and forward loss computation.
    Reads configuration directly from a TrainConfig instance.
    """
    def __init__(self, config: Optional[TrainConfig] = None):
        super().__init__()
        self.config = config if config is not None else TrainConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Base Chatterbox-Flash Pipeline
        model_dir = self.config.model_dir
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            logger.info(f"Loading base Chatterbox-Flash pipeline from local directory: '{model_dir}'...")
            self.tts_engine = ChatterboxFlashTTS.from_pretrained(model_dir, device=self.device)
        else:
            logger.info(f"Local directory '{model_dir}' not found or empty. Downloading from Hugging Face Hub...")
            self.tts_engine = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device=self.device)

        # Freeze Voice Encoder (VE) and S3Gen Vocoder (Only T3 Flash Backbone is trained)
        self.tts_engine.ve.eval()
        self.tts_engine.s3gen.eval()
        for param in self.tts_engine.ve.parameters():
            param.requires_grad = False
        for param in self.tts_engine.s3gen.parameters():
            param.requires_grad = False

        # Extract T3 Flash LLaMA Backbone
        self.t3 = self.tts_engine.t3

        # 2. Apply LoRA or Full Fine-Tuning Setup using TrainConfig
        if self.config.use_lora:
            logger.info(
                f"Setting up LoRA configuration (r={self.config.lora_r}, "
                f"alpha={self.config.lora_alpha}, dropout={self.config.lora_dropout})..."
            )
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                modules_to_save=self.config.lora_modules_to_save
            )
            self.t3 = get_peft_model(self.t3, peft_config)
            self.t3.print_trainable_parameters()
        else:
            logger.info("Configuring model for Full Fine-Tuning (All parameters trainable)...")
            for param in self.t3.parameters():
                param.requires_grad = True

        # Loss Function for Masked Token Prediction (-100 ignores non-masked / padded tokens)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Block-Diffusion training.

        Batch Expected Keys:
        - text_tokens: [B, text_seq_len]
        - prompt_tokens: [B, prompt_seq_len]
        - masked_speech_tokens: [B, speech_seq_len] (contains [MASK] tokens)
        - speaker_emb: [B, spk_dim]
        - labels: [B, speech_seq_len] (-100 for non-masked tokens)
        """
        text_tokens = batch["text_tokens"].to(self.device)
        prompt_tokens = batch["prompt_tokens"].to(self.device)
        masked_speech_tokens = batch["masked_speech_tokens"].to(self.device)
        speaker_emb = batch["speaker_emb"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass through T3 Flash LLaMA backbone
        # Returns logits over speech vocabulary: [B, speech_seq_len, vocab_size]
        logits = self.t3(
            text_tokens=text_tokens,
            prompt_tokens=prompt_tokens,
            speech_tokens=masked_speech_tokens,
            speaker_emb=speaker_emb
        )

        # Reshape logits and labels for CrossEntropyLoss computation
        vocab_size = logits.size(-1)
        loss = self.loss_fn(
            logits.view(-1, vocab_size),
            labels.view(-1)
        )

        return {
            "loss": loss,
            "logits": logits
        }

    def save_checkpoint(self, output_dir: Optional[str] = None):
        """
        Saves trained LoRA weights or Full model checkpoint.
        """
        save_path = output_dir if output_dir is not None else self.config.output_dir
        os.makedirs(save_path, exist_ok=True)

        if self.config.use_lora:
            logger.info(f"Saving LoRA adapter checkpoint to '{save_path}'...")
            self.t3.save_pretrained(save_path)
        else:
            logger.info(f"Saving Full model checkpoint to '{save_path}'...")
            torch.save(self.t3.state_dict(), os.path.join(save_path, "t3_flash_full.pt"))


def build_model_for_training(config: Optional[TrainConfig] = None) -> ChatterboxFlashForTraining:
    """
    Factory function to initialize ChatterboxFlashForTraining from a TrainConfig instance.
    """
    if config is None:
        config = TrainConfig()
    return ChatterboxFlashForTraining(config=config)