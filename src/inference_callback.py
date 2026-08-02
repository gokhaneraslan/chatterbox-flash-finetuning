import os
import logging
import torch
import soundfile as sf
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class InferenceCallback(TrainerCallback):
    """
    HuggingFace Trainer Callback that generates a batch of sample audio utterances during training
    steps to monitor multi-sentence Turkish voice synthesis quality in real-time.
    """
    def __init__(self, tts_engine, config):
        super().__init__()
        self.tts_engine = tts_engine
        self.config = config
        os.makedirs(self.config.sample_output_dir, exist_ok=True)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Trigger evaluation every 'eval_steps'
        if state.global_step > 0 and state.global_step % self.config.eval_steps == 0:
            sample_texts = self.config.eval_sample_texts
            prompt_path = self.config.eval_prompt_path

            if not prompt_path or not os.path.exists(prompt_path):
                logger.warning(f"Evaluation prompt audio '{prompt_path}' not found. Skipping audio generation callback.")
                return

            if not sample_texts or len(sample_texts) == 0:
                logger.warning("No evaluation texts provided in config. Skipping callback.")
                return

            logger.info(f"[Step {state.global_step}] Generating {len(sample_texts)} validation audio samples...")

            try:
                # 1. Put model in eval mode temporarily
                self.tts_engine.t3.eval()

                with torch.inference_mode():
                    # 2. Extract reference speaker conditioning ONCE for all evaluation sentences
                    conds = self.tts_engine.prepare_conditionals(prompt_path)
                    conds_list = [conds] * len(sample_texts)

                    # 3. Generate audio in batch using official Chatterbox-Flash generate_batch API
                    wavs = self.tts_engine.generate_batch(
                        texts=sample_texts,
                        conds_list=conds_list,
                        normalize_text=False  # Text is already normalized / mapped
                    )

                # 4. Save each generated waveform to output directory
                step_dir = os.path.join(self.config.sample_output_dir, f"step_{state.global_step}")
                os.makedirs(step_dir, exist_ok=True)

                for idx, (text, wav) in enumerate(zip(sample_texts, wavs)):
                    out_path = os.path.join(step_dir, f"sample_{idx:02d}.wav")
                    # Convert torch tensor to numpy for soundfile saving
                    wav_np = wav.detach().float().cpu().numpy().squeeze()
                    sf.write(out_path, wav_np, self.tts_engine.sr)
                    logger.info(f"   Saved sample {idx+1}/{len(sample_texts)}: '{out_path}'")

            except Exception as e:
                logger.error(f"Error during validation audio callback at step {state.global_step}: {e}")

            finally:
                # 5. Restore training mode for backbone
                self.tts_engine.t3.train()