import os
import logging
import torch
import numpy as np
import soundfile as sf
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class InferenceCallback(TrainerCallback):

    def __init__(self, tts_engine, config):
        super().__init__()
        self.tts_engine = tts_engine
        self.config = config
        os.makedirs(self.config.sample_output_dir, exist_ok=True)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
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
                self.tts_engine.t3.eval()

                with torch.inference_mode():
                    conds = self.tts_engine.prepare_conditionals(prompt_path)
                    conds_list = [conds] * len(sample_texts)

                    wavs = self.tts_engine.generate_batch(
                        texts=sample_texts,
                        conds_list=conds_list,
                        normalize_text=False
                    )

                step_dir = os.path.join(self.config.sample_output_dir, f"step_{state.global_step}")
                os.makedirs(step_dir, exist_ok=True)

                for idx, (text, wav) in enumerate(zip(sample_texts, wavs)):
                    out_path = os.path.join(step_dir, f"sample_{idx:02d}.wav")
                    wav_np = wav.detach().float().cpu().numpy().squeeze()
                    max_val = np.max(np.abs(wav_np))
                    if max_val > 1.0:
                        wav_np = wav_np / max_val  # Peak Normalization
                    sf.write(out_path, wav_np, self.tts_engine.sr)
                    logger.info(f"   Saved sample {idx+1}/{len(sample_texts)}: '{out_path}'")

            except Exception as e:
                logger.error(f"Error during validation audio callback at step {state.global_step}: {e}")

            finally:
                self.tts_engine.t3.train()