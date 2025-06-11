import torch.nn as nn
import torch

class Eve(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.seq_len , cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.seq_len),
            nn.Sigmoid()
        )

    def forward(self, ciphertext):
        x = torch.cat([ciphertext], dim=-1)
        return self.net(x)
