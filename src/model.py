import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict

from peft import LoraConfig, get_peft_model
from src._chatterbox_flash._chatterbox.models.t3.modules.cond_enc import T3Cond
from src._chatterbox_flash import ChatterboxFlashTTS
from src.config import TrainConfig

logger = logging.getLogger(__name__)

def create_t3_block_causal_attention_mask(
    batch_size,
    len_cond, 
    len_text, 
    len_speech, 
    block_size,
    text_token_lens, 
    speech_token_lens, 
    device, 
    dtype
):
    total_len = len_cond + len_text + len_speech
    prefix_len = len_cond + len_text

    valid = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
    for i in range(batch_size):
        valid[i, :len_cond] = True
        ttl = min(int(text_token_lens[i]), len_text)
        valid[i, len_cond:len_cond + ttl] = True
        stl = min(int(speech_token_lens[i]), len_speech)
        valid[i, prefix_len:prefix_len + stl] = True

    allow = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
    prefix_idx = torch.arange(prefix_len, device=device)
    allow[:prefix_len, :prefix_len] = prefix_idx[:, None] >= prefix_idx[None, :]   # prefix: causal

    speech_block_id = torch.arange(len_speech, device=device) // block_size
    allow[prefix_len:, :prefix_len] = True                                         # speech -> tüm prefix
    allow[prefix_len:, prefix_len:] = speech_block_id[:, None] >= speech_block_id[None, :]  # blok-nedensel

    neg_inf = torch.finfo(dtype).min
    key_valid = valid[:, None, :]
    combined = allow[None, :, :] & key_valid
    additive = torch.zeros((batch_size, total_len, total_len), dtype=dtype, device=device)
    additive.masked_fill_(~combined, neg_inf)
    return additive[:, None, :, :]

class ChatterboxFlashForTraining(nn.Module):
    
    def __init__(self, config: Optional[TrainConfig] = None):
        super().__init__()
        self.config = config if config is not None else TrainConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = self.config.model_dir
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            logger.info(f"Loading base Chatterbox-Flash pipeline from local directory: '{model_dir}'...")
            self.tts_engine = ChatterboxFlashTTS.from_local(model_dir, device=self.device)
        else:
            logger.info(f"Local directory '{model_dir}' not found or empty. Downloading from Hugging Face Hub...")
            self.tts_engine = ChatterboxFlashTTS.from_pretrained("ResembleAI/chatterbox-flash", device=self.device)

        self.tts_engine.ve.eval()
        self.tts_engine.s3gen.eval()
        for param in self.tts_engine.ve.parameters():
            param.requires_grad = False
        for param in self.tts_engine.s3gen.parameters():
            param.requires_grad = False


        self.t3 = self.tts_engine.t3
        self._patch_t3_forward()

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
            if hasattr(self.t3, "enable_input_require_grads"):
                self.t3.enable_input_require_grads()
            self.t3.print_trainable_parameters()
        else:
            logger.info("Configuring model for Full Fine-Tuning (All parameters trainable)...")
            for param in self.t3.parameters():
                param.requires_grad = True

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _patch_t3_forward(self):

        original_t3 = self.t3
        def patched_forward(
            t3_cond,
            text_tokens,
            text_token_lens,
            speech_tokens,
            speech_token_lens,
            training=True,
            is_uncond=None,
            **kwargs
        ):

            embeds, len_cond = original_t3.prepare_input_embeds(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                speech_tokens=speech_tokens,
            )

            batch_size = text_tokens.size(0)
            len_text = text_tokens.size(1)
            len_speech = speech_tokens.size(1)

            if is_uncond is not None and is_uncond.any():
                idx = is_uncond.nonzero(as_tuple=True)[0]
                embeds = embeds.clone()
                embeds[idx, :len_cond, :] = 0.0
                text_positions = torch.arange(len_text, device=embeds.device)[None, :]
                text_valid = text_positions < text_token_lens[idx][:, None]
                text_slice = embeds[idx, len_cond:len_cond + len_text, :]
                embeds[idx, len_cond:len_cond + len_text, :] = torch.where(
                    text_valid.unsqueeze(-1), torch.zeros_like(text_slice), text_slice,
                )

            attn_mask = create_t3_block_causal_attention_mask(
                batch_size=batch_size,
                len_cond=len_cond,
                len_text=len_text,
                len_speech=len_speech,
                block_size=self.config.block_size,
                text_token_lens=text_token_lens,
                speech_token_lens=speech_token_lens,
                device=embeds.device,
                dtype=embeds.dtype,
            )

            tfmr_out = original_t3.tfmr.forward(
                input_ids=None,
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=(not training),
            )

            hidden_states = tfmr_out.hidden_states[-1]

            B, _, dim = hidden_states.shape
            device, dtype = hidden_states.device, hidden_states.dtype
            text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
            speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
            ttl, stl = text_token_lens, speech_token_lens

            for i in range(B):
                text_end = len_cond + ttl[i].item()
                speech_start = len_cond + len_text
                speech_end = speech_start + stl[i].item()
                text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
                speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

            text_logits = original_t3.text_head(text_latents)
            speech_logits = original_t3.speech_head(speech_latents)

            class T3OutputContainer:
                def __init__(self, speech_logits, text_logits, hidden_states):
                    self.speech_logits = speech_logits
                    self.text_logits = text_logits
                    self.hidden_states = hidden_states

            return T3OutputContainer(
                speech_logits=speech_logits,
                text_logits=text_logits,
                hidden_states=hidden_states
            )

        self.t3.forward = patched_forward

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.t3, "gradient_checkpointing_enable"):
            self.t3.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        text_token_lens: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        prompt_token_lens: Optional[torch.Tensor] = None,
        masked_speech_tokens: Optional[torch.Tensor] = None,
        speech_token_lens: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        model_dtype = next(self.t3.parameters()).dtype

        text_tokens = text_tokens.to(self.device)
        prompt_tokens = prompt_tokens.to(self.device)
        masked_speech_tokens = masked_speech_tokens.to(self.device)
        labels = labels.to(self.device)

        batch_size = text_tokens.size(0)

        speaker_emb = speaker_emb.to(device=self.device, dtype=model_dtype)
        emotion_adv = 0.5 * torch.ones(batch_size, 1, 1, device=self.device, dtype=model_dtype)

        text_token_lens = text_token_lens.to(self.device) if text_token_lens is not None \
            else torch.full((batch_size,), text_tokens.size(1), dtype=torch.long, device=self.device)
        speech_token_lens = speech_token_lens.to(self.device) if speech_token_lens is not None \
            else torch.full((batch_size,), masked_speech_tokens.size(1), dtype=torch.long, device=self.device)
        prompt_token_lens = prompt_token_lens.to(self.device) if prompt_token_lens is not None \
            else torch.full((batch_size,), prompt_tokens.size(1), dtype=torch.long, device=self.device)

        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=prompt_tokens,
            cond_prompt_speech_lens=prompt_token_lens,
            emotion_adv=emotion_adv
        ).to(device=self.device, dtype=model_dtype)

        is_uncond = kwargs.get("is_uncond")
        if is_uncond is not None:
            is_uncond = is_uncond.to(self.device)
        
        out = self.t3(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=masked_speech_tokens,
            speech_token_lens=speech_token_lens,
            is_uncond=is_uncond
        )

        if hasattr(out, "speech_logits"):
            logits = out.speech_logits
        elif hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, torch.Tensor):
            logits = out
        elif isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out

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

        save_path = output_dir if output_dir is not None else self.config.output_dir
        os.makedirs(save_path, exist_ok=True)

        if self.config.use_lora:
            logger.info(f"Saving LoRA adapter checkpoint to '{save_path}'...")
            self.t3.save_pretrained(save_path)
        else:
            logger.info(f"Saving Full model checkpoint to '{save_path}'...")
            torch.save(self.t3.state_dict(), os.path.join(save_path, "t3_flash_full.pt"))


def build_model_for_training(config: Optional[TrainConfig] = None) -> ChatterboxFlashForTraining:
    if config is None:
        config = TrainConfig()
    return ChatterboxFlashForTraining(config=config)