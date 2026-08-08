import os
import random
import torch
import logging
from torch.utils.data import Dataset
from src.config import TrainConfig

logger = logging.getLogger(__name__)


cfg = TrainConfig()

class ChatterboxFlashDataset(Dataset):

    def __init__(
        self,
        processed_dir: str = cfg.preprocessed_dir,
        max_text_len: int = cfg.max_text_len,
        max_speech_len: int = cfg.max_speech_len,
        uncond_prob: float = cfg.uncond_prob,
        sot_token: int = cfg.start_text_token,
        eot_token: int = cfg.stop_text_token,
        speech_stop_id: int = cfg.speech_stop_id,
        mask_token_id: int = cfg.mask_token_id
    ):
        super().__init__()
        self.processed_dir = processed_dir
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        self.uncond_prob = uncond_prob
        self.sot_token = sot_token
        self.eot_token = eot_token
        self.speech_stop_id = speech_stop_id
        self.mask_token_id = mask_token_id

        if not os.path.exists(self.processed_dir):
            raise FileNotFoundError(f"Processed feature directory not found: '{self.processed_dir}'")

        self.files = [f for f in os.listdir(self.processed_dir) if f.endswith(".pt")]
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt feature files found in: '{self.processed_dir}'")

        logger.info(f"ChatterboxFlashDataset initialized with {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        pt_path = os.path.join(self.processed_dir, filename)

        try:
            data = torch.load(pt_path, map_location="cpu")

            text_tokens = data["text_tokens"]
            speech_tokens = data["speech_tokens"]
            speaker_emb = data["speaker_emb"]
            prompt_tokens = data["prompt_tokens"]

            if text_tokens.size(0) > self.max_text_len - 2:
                text_tokens = text_tokens[: self.max_text_len - 2]

            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            if speech_tokens.size(0) > self.max_speech_len:
                speech_tokens = speech_tokens[: self.max_speech_len - 1]
                stop_tensor = torch.tensor([self.speech_stop_id], dtype=torch.long)
                speech_tokens = torch.cat([speech_tokens, stop_tensor])
            elif speech_tokens[-1].item() != self.speech_stop_id:
                stop_tensor = torch.tensor([self.speech_stop_id], dtype=torch.long)
                speech_tokens = torch.cat([speech_tokens, stop_tensor])

            if random.random() < self.uncond_prob:
                speaker_emb = torch.zeros_like(speaker_emb)
                prompt_tokens = torch.full((1,), self.mask_token_id, dtype=torch.long)

            return {
                "id": filename.replace(".pt", ""),
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens
            }

        except Exception as e:
            logger.error(f"Error loading sample '{filename}': {e}")
            return None