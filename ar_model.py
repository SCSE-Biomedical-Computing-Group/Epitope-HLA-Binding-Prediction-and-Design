"""
Transformer-based autoregressive model for peptide generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# Standard 20 amino acids + special tokens
AA_VOCAB = ['<PAD>', '<BOS>', '<EOS>'] + list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_VOCAB)}
VOCAB_SIZE = len(AA_VOCAB)
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


class PeptideEmbedding(nn.Module):
    """Embedding stack: token + learned position + layer norm + dropout."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_IDX,
        )
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids)
        x = x + self.pos_embed(positions)
        x = self.layer_norm(x)
        return self.dropout(x)


class AlleleEncoder(nn.Module):
    """
    Encodes HLA allele strings into embeddings.
    Uses character-level encoding of allele name.
    """

    def __init__(self, embed_dim: int, max_allele_len: int = 20):
        super().__init__()
        # Simple character vocabulary for allele names
        self.chars = '<PAD>ABCDEFGHIJKLMNOPQRSTUVWXYZ*:-0123456789'
        self.char_vocab = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.max_len = max_allele_len

        self.char_embed = nn.Embedding(
            self.vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_embed = nn.Embedding(max_allele_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def encode_allele(self, allele: str) -> List[int]:
        """Convert allele string to character indices."""
        indices = []
        for c in allele.upper()[:self.max_len]:
            # 0 = PAD for unknown chars
            indices.append(self.char_vocab.get(c, 0))
        # Pad to max_len
        while len(indices) < self.max_len:
            indices.append(0)
        return indices

    def forward(
        self,
        alleles: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        # Encode all alleles
        encoded = [self.encode_allele(a) for a in alleles]
        token_ids = torch.tensor(
            encoded,
            dtype=torch.long,
            device=device,
        )  # (batch, max_len)

        positions = torch.arange(self.max_len, device=device).unsqueeze(0)
        x = self.char_embed(token_ids) + self.pos_embed(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Masked mean pooling to get a fixed-size allele embedding.
        mask = token_ids.ne(PAD_IDX).unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.proj(pooled)


class LengthEncoder(nn.Module):
    """Encodes target peptide length (8-11) into embeddings."""

    def __init__(self, embed_dim: int, min_len: int = 8, max_len: int = 11):
        super().__init__()
        self.min_len = min_len
        self.max_len = max_len
        self.num_lengths = max_len - min_len + 1
        self.embed = nn.Embedding(self.num_lengths, embed_dim)

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        idx = lengths - self.min_len
        idx = idx.clamp(0, self.num_lengths - 1)
        return self.embed(idx)


class TransformerARDecoder(nn.Module):
    """
    Small Transformer decoder for autoregressive peptide generation.
    Uses a shared backbone with a single class-I output head.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 14,  # BOS + 11 + EOS + buffer
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Peptide embedding stack
        self.peptide_embed = PeptideEmbedding(
            vocab_size=VOCAB_SIZE,
            embed_dim=embed_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # Conditioning encoders
        self.allele_encoder = AlleleEncoder(embed_dim)
        self.length_encoder = LengthEncoder(embed_dim)

        # Conditioning projection (allele + length -> context)
        self.cond_proj = nn.Linear(embed_dim * 2, embed_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_proj = nn.Linear(embed_dim, VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(
        self,
        sz: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        alleles: List[str],
        lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass for training."""
        _, seq_len = tokens.shape

        # Get conditioning embeddings
        allele_emb = self.allele_encoder(alleles, device)  # (batch, embed_dim)
        length_emb = self.length_encoder(lengths)          # (batch, embed_dim)
        # (batch, embed_dim)
        cond = self.cond_proj(torch.cat([allele_emb, length_emb], dim=-1))

        # Token + positional embeddings
        x = self.peptide_embed(tokens)

        # Add conditioning as first "memory" token
        memory = cond.unsqueeze(1)  # (batch, 1, embed_dim)

        # Generate causal mask
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device)

        # Transformer forward
        # (batch, seq_len, embed_dim)
        out = self.transformer(x, memory, tgt_mask=tgt_mask)

        return self.output_proj(out)

    @torch.no_grad()
    def generate(
        self,
        alleles: List[str],
        lengths: List[int],
        device: torch.device,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> Tuple[List[str], torch.Tensor]:
        """Generate peptides autoregressively."""
        self.eval()
        batch_size = len(alleles)
        target_lengths = [
            int(length.item()) if torch.is_tensor(length) else int(length)
            for length in lengths
        ]
        lengths_t = torch.tensor(
            target_lengths,
            dtype=torch.long,
            device=device,
        )

        # Get conditioning
        allele_emb = self.allele_encoder(alleles, device)
        length_emb = self.length_encoder(lengths_t)
        cond = self.cond_proj(torch.cat([allele_emb, length_emb], dim=-1))
        memory = cond.unsqueeze(1)

        # Start with BOS token
        tokens = torch.full(
            (batch_size, 1),
            BOS_IDX,
            dtype=torch.long,
            device=device,
        )
        log_probs = torch.zeros(batch_size, device=device)

        max_len = max(target_lengths) + 1  # +1 for potential EOS

        for step in range(max_len):
            # Embed current sequence
            x = self.peptide_embed(tokens)

            # Get causal mask
            tgt_mask = self._generate_square_subsequent_mask(
                tokens.size(1),
                device,
            )

            # Forward through transformer
            out = self.transformer(x, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(out[:, -1, :])  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Mask special tokens (except for length check)
            logits[:, PAD_IDX] = -float('inf')
            # Allow EOS only after reaching target length
            for i in range(batch_size):
                if step < target_lengths[i]:
                    logits[i, EOS_IDX] = -float('inf')
                    logits[i, BOS_IDX] = -float('inf')

            # Apply top-k if specified
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[
                    0][..., -1, None]
                logits[indices_to_remove] = -float('inf')

            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )
                sorted_indices_to_remove[..., 0] = False

                for i in range(batch_size):
                    indices_to_remove = (
                        sorted_indices[i][sorted_indices_to_remove[i]]
                    )
                    logits[i, indices_to_remove] = -float('inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Update log probs
            for i in range(batch_size):
                if step < target_lengths[i]:
                    log_probs[i] += torch.log(
                        probs[i, next_token[i, 0]] + 1e-10
                    )

            # Append token
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check if all sequences reached target length
            if all(
                tokens.size(1) - 1 >= target_lengths[i]
                for i in range(batch_size)
            ):
                break

        # Convert tokens to peptide strings
        peptides = []
        for i in range(batch_size):
            # Skip BOS, exact length
            seq_tokens = tokens[i, 1:target_lengths[i] + 1]
            token_ids = [int(token.item()) for token in seq_tokens]
            peptide = ''.join(
                IDX_TO_AA.get(token_id, '')
                for token_id in token_ids
                if token_id not in [PAD_IDX, BOS_IDX, EOS_IDX]
            )
            peptides.append(peptide)

        return peptides, log_probs


def create_ar_model(embed_dim: int = 128, **kwargs) -> nn.Module:
    """Create the autoregressive Transformer decoder model."""
    return TransformerARDecoder(embed_dim=embed_dim, **kwargs)


def encode_peptide(peptide: str) -> List[int]:
    """Convert peptide string to token indices with BOS prefix."""
    tokens = [BOS_IDX]
    for aa in peptide:
        if aa in AA_TO_IDX:
            tokens.append(AA_TO_IDX[aa])
    return tokens
