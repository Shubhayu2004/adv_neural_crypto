import torch
import torch.nn as nn

class TransformerAlice(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.input_dim = 2  # plaintext + key (bit per position)
        self.embed_dim = cfg.transformer_embed_dim
        self.nhead = cfg.transformer_nhead
        self.num_layers = cfg.transformer_num_layers
        self.hidden_dim = cfg.transformer_hidden_dim

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.nhead, dim_feedforward=self.hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_proj = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pt, key):
        # pt, key: (batch, seq_len)
        x = torch.stack([pt, key], dim=-1)  # (batch, seq_len, 2)
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        x = self.output_proj(x).squeeze(-1)  # (batch, seq_len)
        return self.sigmoid(x)

class TransformerBob(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.input_dim = 2  # ciphertext + key
        self.embed_dim = cfg.transformer_embed_dim
        self.nhead = cfg.transformer_nhead
        self.num_layers = cfg.transformer_num_layers
        self.hidden_dim = cfg.transformer_hidden_dim

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.nhead, dim_feedforward=self.hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_proj = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ct, key):
        # ct, key: (batch, seq_len)
        x = torch.stack([ct, key], dim=-1)  # (batch, seq_len, 2)
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        x = self.output_proj(x).squeeze(-1)  # (batch, seq_len)
        return self.sigmoid(x)

class TransformerEve(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.input_dim = 1  # ciphertext only
        self.embed_dim = cfg.transformer_embed_dim
        self.nhead = cfg.transformer_nhead
        self.num_layers = cfg.transformer_num_layers
        self.hidden_dim = cfg.transformer_hidden_dim

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.nhead, dim_feedforward=self.hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_proj = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ct):
        # ct: (batch, seq_len)
        x = ct.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        x = self.output_proj(x).squeeze(-1)  # (batch, seq_len)
        return self.sigmoid(x)
