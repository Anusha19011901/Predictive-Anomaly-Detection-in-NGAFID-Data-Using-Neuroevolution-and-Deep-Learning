import math, torch, torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d]

    def forward(self, x):  # x: [b, L, d]
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesBERT(nn.Module):
    """
    BERT-style encoder for continuous time series:
    - Linear patch embedding per time step
    - Learned [MASK] token vector
    - Transformer encoder
    - Heads:
      * pretraining: predict masked values (regression)
      * finetune: binary classification (anomaly vs normal)
    """
    def __init__(self, d_in, d_model, n_layers, n_heads, ff_mult=4, dropout=0.1, max_len=4096):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*ff_mult,
            batch_first=True, dropout=dropout, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # heads
        self.to_value = nn.Linear(d_model, d_in)     # for masked value regression (MSE)
        self.cls_head = nn.Sequential(               # for fine-tune classification
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, attn_mask=None):
        # x: [b, L, d_in]
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h, mask=None)  # causal mask not used; BERT is bidirectional
        return h

    def predict_masked(self, x_filled, mask_bool):
        # x_filled: inputs with masked positions already replaced by mask_token-projected
        h = self.forward(x_filled)
        y_pred = self.to_value(h)  # [b, L, d_in]
        return y_pred[mask_bool]   # only compute loss on masked positions

    def classify(self, x):
        h = self.forward(x)         # [b, L, d_model]
        pooled = h.mean(dim=1)      # simple mean pool
        return self.cls_head(pooled).squeeze(-1)  # logits
