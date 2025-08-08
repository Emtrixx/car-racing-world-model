class DIY_TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # --- Sub-layer 1: Self-Attention on the target (queries) ---
        # Output queries talk to each other.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # --- Sub-layer 2: Cross-Attention (target queries to encoder memory) ---
        # The queries look at the historical context.
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # --- Sub-layer 3: Feed-Forward Network ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # --- Self-Attention Block ---
        # The queries attend to themselves.
        attn_output, _ = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        # Add & Norm
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # --- Cross-Attention Block ---
        # The queries from the previous block now attend to the full memory.
        attn_output, _ = self.multihead_attn(query=tgt, key=memory, value=memory, attn_mask=memory_mask)
        # Add & Norm
        tgt = self.norm2(tgt + self.dropout2(attn_output))

        # --- Feed-Forward Block ---
        ffn_output = self.ffn(tgt)
        # Add & Norm
        tgt = self.norm3(tgt + self.dropout3(ffn_output))

        return tgt