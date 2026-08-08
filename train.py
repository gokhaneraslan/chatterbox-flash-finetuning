import os
import sys
import logging
import warnings
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TrainConfig
from src.dataset import ChatterboxFlashDataset
from src.collator import FlashDataCollator
from src.model import build_model_for_training
from src.inference_callback import InferenceCallback
from src.preprocess import preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ChatterboxFlashTrain")


def main():


    cfg = TrainConfig()

    logger.info("==================================================")
    logger.info("   Starting Chatterbox-Flash Fine-Tuning Pipeline ")
    logger.info("==================================================")
    logger.info(f"Training Mode: {'LoRA' if cfg.use_lora else 'Full Fine-Tune'}")
    logger.info(f"Model Directory: '{cfg.model_dir}'")
    logger.info(f"Output Directory: '{cfg.output_dir}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    if not os.path.exists(cfg.preprocessed_dir) or len(os.listdir(cfg.preprocessed_dir)) == 0:
        logger.info("Preprocessed feature directory empty. Running preprocessing step...")
        preprocess(
            metadata_path=cfg.metadata_path,
            wav_dir=cfg.wav_dir,
            output_dir=cfg.preprocessed_dir,
            model_dir=cfg.model_dir,
            prompt_duration=cfg.prompt_duration,
            use_vad=cfg.use_vad
        )
    else:
        logger.info(f"Preprocessed features found in '{cfg.preprocessed_dir}'. Skipping preprocessing step.")


    logger.info("Initializing Chatterbox-Flash model wrapper for training...")
    model_wrapper = build_model_for_training(cfg)

    logger.info("Initializing Dataset and Flash Block-Diffusion Data Collator...")
    train_dataset = ChatterboxFlashDataset(
        processed_dir=cfg.preprocessed_dir,
        max_text_len=cfg.max_text_len,
        max_speech_len=cfg.max_speech_len,
        uncond_prob=cfg.uncond_prob
    )

    data_collator = FlashDataCollator(
        mask_token_id=cfg.mask_token_id,
        pad_token_id=cfg.pad_token_id,
        block_size=cfg.block_size,
        min_mask_ratio=cfg.min_mask_ratio,
        max_mask_ratio=cfg.max_mask_ratio
    )

    callbacks = []
    if cfg.eval_prompt_path and os.path.exists(cfg.eval_prompt_path):
        logger.info(f"Audio generation callback enabled. Samples will be saved every {cfg.eval_steps} steps.")
        inference_cb = InferenceCallback(model_wrapper.tts_engine, cfg)
        callbacks.append(inference_cb)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_epochs,
        warmup_steps=cfg.warmup_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,  # Required for custom dict batch collator
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=(cfg.mixed_precision == "fp16"),
        bf16=(cfg.mixed_precision == "bf16"),
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=True,  # Reduces VRAM consumption significantly
        dataloader_persistent_workers=(cfg.dataloader_num_workers > 0),
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )

    logger.info("Starting training loop...")
    trainer.train()

    logger.info("Training complete. Saving final model weights...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.use_lora:
        save_path = os.path.join(cfg.output_dir, "chatterbox_flash_tr_lora")
        model_wrapper.t3.save_pretrained(save_path)
        logger.info(f"LoRA adapter saved successfully to: '{save_path}'")
    else:
        final_model_path = os.path.join(cfg.output_dir, "t3_flash_tr_full.safetensors")
        save_file(model_wrapper.t3.state_dict(), final_model_path)
        logger.info(f"Full model checkpoint saved successfully to: '{final_model_path}'")

if __name__ == "__main__":
    main()