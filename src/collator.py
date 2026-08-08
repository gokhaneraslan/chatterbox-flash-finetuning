import torch
import random
from torch.nn.utils.rnn import pad_sequence
from src.config import TrainConfig

cfg = TrainConfig()

class FlashDataCollator:

    def __init__(
        self,
        mask_token_id: int = cfg.mask_token_id,
        pad_token_id: int = cfg.pad_token_id,
        block_size: int = cfg.block_size,
        min_mask_ratio: float = cfg.min_mask_ratio,
        max_mask_ratio: float = cfg.max_mask_ratio
    ):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.block_size = block_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def __call__(self, batch):

        batch = [item for item in batch if item is not None]
        if not batch:
            return {}

        text_tokens = pad_sequence(
            [x["text_tokens"] for x in batch],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        text_token_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)

        prompt_tokens = pad_sequence(
            [x["prompt_tokens"] for x in batch],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        prompt_token_lens = torch.tensor([len(x["prompt_tokens"]) for x in batch], dtype=torch.long)

        clean_speech_tokens = pad_sequence(
            [x["speech_tokens"] for x in batch],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        speech_token_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

        speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

        masked_speech_tokens = clean_speech_tokens.clone()
        labels = torch.full_like(clean_speech_tokens, fill_value=-100)  # Default ignore_index for CrossEntropyLoss

        batch_size, max_speech_seq_len = clean_speech_tokens.shape

        for i in range(batch_size):
            seq_len = speech_token_lens[i].item()
            if seq_len == 0:
                continue

            mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
            target_mask_count = max(1, int(seq_len * mask_ratio))

            mask_bool = torch.zeros(seq_len, dtype=torch.bool)
            num_masked = 0
            attempts = 0
            max_attempts = 50

            while num_masked < target_mask_count and attempts < max_attempts:
                attempts += 1
                start_idx = random.randint(0, max(0, seq_len - 1))
                end_idx = min(seq_len, start_idx + self.block_size)
                
                mask_bool[start_idx:end_idx] = True
                num_masked = mask_bool.sum().item()

            mask_indices = torch.where(mask_bool)[0]
            masked_speech_tokens[i, mask_indices] = self.mask_token_id
            labels[i, mask_indices] = clean_speech_tokens[i, mask_indices]

        return {
            "text_tokens": text_tokens,
            "text_token_lens": text_token_lens,
            "prompt_tokens": prompt_tokens,
            "prompt_token_lens": prompt_token_lens,
            "masked_speech_tokens": masked_speech_tokens,
            "labels": labels,
            "speech_token_lens": speech_token_lens,
            "speaker_emb": speaker_embs
        }