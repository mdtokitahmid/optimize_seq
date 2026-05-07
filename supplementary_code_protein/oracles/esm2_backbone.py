"""
ESM2SoftBackbone — differentiable soft-input ESM-2 wrapper.

Bypasses the embeddings layer entirely for soft_forward, calling the
transformer encoder directly with pre-built embeddings. This avoids all
masked_fill / input_ids issues in HuggingFace's EsmEmbeddings.forward.
"""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from typing import Optional

from utils.encoding import AA_ALPHABET


class ESM2SoftBackbone(nn.Module):

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.tokenizer  = EsmTokenizer.from_pretrained(model_name)
        self.esm        = EsmModel.from_pretrained(model_name)
        self.d_model    = self.esm.config.hidden_size

        if freeze:
            for p in self.esm.parameters():
                p.requires_grad_(False)

        self.aa_token_ids = self._get_aa_token_ids()
        self.register_buffer("aa_embed_matrix", self._build_aa_embed_matrix())

    def _get_aa_token_ids(self) -> torch.Tensor:
        ids = []
        for aa in AA_ALPHABET:
            tok_id = self.tokenizer.convert_tokens_to_ids(aa)
            if tok_id == self.tokenizer.unk_token_id:
                raise ValueError(f"ESM-2 tokenizer does not know AA '{aa}'")
            ids.append(tok_id)
        return torch.tensor(ids, dtype=torch.long)

    def _build_aa_embed_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            W = self.esm.embeddings.word_embeddings.weight
            return W[self.aa_token_ids].clone()

    def soft_forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable forward pass for soft one-hot inputs (B, 20, L).

        Instead of going through self.esm() (which calls the embeddings layer
        and crashes on soft input), we:
          1. Build the full embedding tensor manually (BOS + soft AAs + EOS)
          2. Call the transformer encoder directly, bypassing the embeddings layer

        This is equivalent to what ESM does internally, minus the token lookup
        and the masked_fill that causes the crash.

        Returns: (B, L+2, d_model)
        """
        B, K, L = x.shape
        assert K == 20, f"Expected 20 AA channels, got {K}"

        # 1. Weighted-sum embedding: (B,L,20) @ (20,d) -> (B,L,d)
        soft_embeds = torch.matmul(x.permute(0, 2, 1), self.aa_embed_matrix)

        # 2. BOS and EOS embeddings from the word embedding table
        W = self.esm.embeddings.word_embeddings.weight.detach()
        bos = W[self.tokenizer.cls_token_id].view(1, 1, -1).expand(B, 1, -1)
        eos = W[self.tokenizer.eos_token_id].view(1, 1, -1).expand(B, 1, -1)
        full_embeds = torch.cat([bos, soft_embeds, eos], dim=1)  # (B, L+2, d)

        # 3. Token-dropout scaling — ESM2 trains with token_dropout=True, which at
        #    inference (no mask tokens) scales all word embeddings by (1 - 0.15*0.8) = 0.88.
        #    Skipping this makes soft_forward embeddings ~14% too large vs. what the
        #    fine-tuned oracle head was trained on, causing wrong oracle scores.
        if self.esm.embeddings.token_dropout:
            full_embeds = full_embeds * (1.0 - 0.15 * 0.8)

        # 4. Add position embeddings (applied after token-dropout scaling, matching ESM forward)
        seq_len = full_embeds.shape[1]  # L+2
        position_ids = torch.arange(1, seq_len + 1, dtype=torch.long,
                                    device=x.device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.esm.embeddings.position_embeddings(position_ids)
        full_embeds = full_embeds + pos_embeds

        # 5. Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=x.device)

        # 7. Extend attention mask to (B, 1, 1, seq_len) as the encoder expects
        extended_mask = self.esm.get_extended_attention_mask(
            attention_mask, (B, seq_len)
        )

        # 8. Run the transformer encoder directly (skip embeddings layer)
        encoder_outputs = self.esm.encoder(
            full_embeds,
            attention_mask=extended_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state  # (B, L+2, d_model)

    def hard_forward(self, sequences: list) -> torch.Tensor:
        """Standard forward pass from sequence strings."""
        device = self.aa_embed_matrix.device
        inputs = self.tokenizer(sequences, return_tensors="pt",
                                padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.esm(**inputs)
        return outputs.last_hidden_state