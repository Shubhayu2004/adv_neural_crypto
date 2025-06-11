import pytorch_lightning as pl
import torch
import torch.nn as nn
from modules.alice import Alice
from modules.bob import Bob
from modules.eve import Eve
from utils.losses import CompositeLoss

class AdvCryptoModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.alice = Alice(cfg)
        self.bob = Bob(cfg)
        self.eves = nn.ModuleList([Eve(cfg) for _ in range(cfg.num_eves)])
        self.losses = CompositeLoss()

    def forward(self, pt, key):
        ct = self.alice(pt, key)
        pt_hat = self.bob(ct, key)
        return ct, pt_hat

    def training_step(self, batch, batch_idx, optimizer_idx):
        pt, key = batch
        ct, pt_hat = self(pt, key)

        if optimizer_idx == 0:
            loss_ab = self.losses.bob_loss(pt_hat, pt) - self.losses.adversary_loss(ct, pt, self.eves)
            return loss_ab

        else:
            i = optimizer_idx - 1
            pt_rev = self.eves[i](ct)
            return self.losses.eve_loss(pt_rev, pt)

    def configure_optimizers(self):
        opt_ab = torch.optim.Adam(list(self.alice.parameters()) + list(self.bob.parameters()), lr=self.cfg.lr)
        opt_es = [torch.optim.Adam(e.parameters(), lr=self.cfg.lr) for e in self.eves]
        return [opt_ab] + opt_es, []
