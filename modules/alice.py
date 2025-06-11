import torch.nn as nn

class Alice(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.seq_len * 2, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.seq_len),
            nn.Sigmoid()
        )

    def forward(self, plaintext, key):
        x = torch.cat([plaintext, key], dim=-1)
        return self.net(x)
