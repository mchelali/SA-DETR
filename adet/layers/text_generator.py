from typing import Any
from torch import nn
from .pos_encoding import PositionalEncoding1D as PositionalEncoding

class TextGenerator(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers=2, nhead=8, max_len=64, temperature: int = 10000,
    normalize: bool = False,
    scale: Any | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(num_pos_feats=d_model//2, temperature=temperature, normalize=normalize, scale=scale)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, memory, tgt_seq):
        # memory: [batch, num_queries, d_model]
        # tgt_seq: [batch, seq_len] (GT tokens or autoregressive predictions)
        tgt_emb = self.embedding(tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt = tgt_emb.transpose(0, 1)  # [seq_len, batch, d_model]
        memory = memory.transpose(0, 1)  # [num_queries, batch, d_model]
        output = self.decoder(tgt, memory)
        logits = self.output_proj(output.transpose(0, 1))
        return logits
