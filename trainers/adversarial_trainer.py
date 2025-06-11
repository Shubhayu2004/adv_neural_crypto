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
        self.automatic_optimization = False
            
        self.alice = Alice(cfg)
        self.bob = Bob(cfg)
        self.eves = nn.ModuleList([Eve(cfg) for _ in range(cfg.num_eves)])
        self.losses = CompositeLoss()

    def forward(self, pt, key):
        ct = self.alice(pt, key)
        pt_hat = self.bob(ct, key)
        return ct, pt_hat

    def training_step(self, batch):
        pt, key = batch
        ct = self.alice(pt, key)
        pt_hat = self.bob(ct, key)

        opt_ab, *opt_eves = self.optimizers()

        # === Train Alice/Bob ===
        loss_ab = self.losses.bob_loss(pt_hat, pt) - self.losses.adversary_loss(ct, pt, self.eves)
        opt_ab.zero_grad()
        self.manual_backward(loss_ab)
        opt_ab.step()

        # === Train each Eve ===
        for i, eve in enumerate(self.eves):
            pt_pred = eve(ct.detach())
            loss_eve = self.losses.eve_loss(pt_pred, pt)
            opt = opt_eves[i]
            opt.zero_grad()
            self.manual_backward(loss_eve)
            opt.step()

        self.log("loss_ab", loss_ab)
        self.log("loss_eve", loss_eve)

    def configure_optimizers(self):
        opt_ab = torch.optim.Adam(list(self.alice.parameters()) + list(self.bob.parameters()), lr=self.cfg.lr)
        opt_es = [torch.optim.Adam(e.parameters(), lr=self.cfg.lr) for e in self.eves]
        return [opt_ab] + opt_es
