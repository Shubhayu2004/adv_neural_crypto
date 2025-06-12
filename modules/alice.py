import torch
import torch.nn as nn

class Alice(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.seq_len * 2, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.seq_len),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, pt, key):
        x = torch.cat([pt, key], dim=-1)
        return self.net(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
