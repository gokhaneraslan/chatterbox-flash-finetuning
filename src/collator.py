import torch
import random
from torch.nn.utils.rnn import pad_sequence
from src.config import TrainConfig

# Global configuration instance
cfg = TrainConfig()


class FlashDataCollator:
    """
    Data Collator for Chatterbox-Flash Block-Diffusion Masked Training.
    Handles dynamic batch padding and applies random token masking for Block Diffusion.
    """
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
        # Filter out corrupted or None items
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}

        # 1. Pad Text Tokens
        text_tokens = pad_sequence(
            [x["text_tokens"] for x in batch],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)

        # 2. Pad Prompt Tokens
        prompt_tokens = pad_sequence(
            [x["prompt_tokens"] for x in batch],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        prompt_lens = torch.tensor([len(x["prompt_tokens"]) for x in batch], dtype=torch.long)

        # 3. Pad Speech Tokens (Clean Target Speech)
        clean_speech_tokens = pad_sequence(
            [x["speech_tokens"] for x in batch],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

        # 4. Stack Speaker Embeddings
        speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

        # 5. Apply Block-Diffusion Masking to Target Speech Tokens
        masked_speech_tokens = clean_speech_tokens.clone()
        labels = torch.full_like(clean_speech_tokens, fill_value=-100)  # Default ignore_index for CrossEntropyLoss

        batch_size, max_speech_seq_len = clean_speech_tokens.shape

        for i in range(batch_size):
            seq_len = speech_lens[i].item()
            if seq_len == 0:
                continue

            # Pick a random masking ratio for this sample
            mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
            num_mask = max(1, int(seq_len * mask_ratio))

            # Randomly select indices to mask within valid speech length
            mask_indices = torch.randperm(seq_len)[:num_mask]

            # Replace target tokens with [MASK] token
            masked_speech_tokens[i, mask_indices] = self.mask_token_id

            # Set label for loss computation ONLY at masked positions
            labels[i, mask_indices] = clean_speech_tokens[i, mask_indices]

        return {
            "text_tokens": text_tokens,
            "text_lens": text_lens,
            "prompt_tokens": prompt_tokens,
            "prompt_lens": prompt_lens,
            "masked_speech_tokens": masked_speech_tokens,
            "labels": labels,
            "speech_lens": speech_lens,
            "speaker_emb": speaker_embs
        }