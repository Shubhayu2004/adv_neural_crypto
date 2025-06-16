import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.alice import Alice
from modules.bob import Bob
from modules.eve import Eve
from modules.keygen import KeyGen
from utils.losses import CompositeLoss

class AdvCryptoModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.keygen = KeyGen(cfg)
        self.alice = Alice(cfg)
        self.bob = Bob(cfg)
        self.eves = nn.ModuleList([Eve(cfg) for _ in range(cfg.num_eves)])
        self.losses = CompositeLoss()

        self.bob_losses = []
        self.eve_losses = []

    def forward(self, pt, pub_key, priv_key):
        ct = self.alice(pt, pub_key)
        pt_hat = self.bob(ct, priv_key)
        return ct, pt_hat

    def training_step(self, batch, batch_idx):
        pt, _ = batch
        batch_size = pt.shape[0]
        warmup_epochs = 10
        current_epoch = self.trainer.current_epoch

        # Generate public-private key pairs
        # Generate or reuse keys
        if self.cfg.key_reuse:
            if not hasattr(self, "_pub_key") or batch_idx == 0:
                self._pub_key, self._priv_key = self.keygen(batch_size)
            pub_key, priv_key = self._pub_key, self._priv_key
        else:
            pub_key, priv_key = self.keygen(batch_size)

        # Forward pass
        ct = self.alice(pt, pub_key)
        pt_hat = self.bob(ct, priv_key)

        if current_epoch < warmup_epochs:
            # Warm-up: train Alice & Bob only
            opt_ab = self.optimizers()[0]
            true_bob_loss = self.losses.bob_loss(pt_hat, pt)

            opt_ab.zero_grad()
            self.manual_backward(true_bob_loss)
            opt_ab.step()

            self.bob_losses.append(true_bob_loss.item())
            self.eve_losses.append(float('nan'))

        else:
            # Full adversarial training
            opt_ab, *opt_eves = self.optimizers()
            true_bob_loss = self.losses.bob_loss(pt_hat, pt)
            adversarial_penalty = self.losses.adversary_loss_asymmetric(ct, pt, pub_key, self.eves)

            loss_ab = true_bob_loss - adversarial_penalty

            opt_ab.zero_grad()
            self.manual_backward(loss_ab)
            opt_ab.step()

            for i, eve in enumerate(self.eves):
                pt_pred = eve(ct.detach(), pub_key)
                loss_e = self.losses.eve_loss(pt_pred, pt)
                opt = opt_eves[i]
                opt.zero_grad()
                self.manual_backward(loss_e)
                opt.step()

            self.bob_losses.append(true_bob_loss.item())
            avg_eve_loss = sum(F.mse_loss(e(ct, pub_key), pt).item() for e in self.eves) / len(self.eves)
            self.eve_losses.append(avg_eve_loss)

        self.log("loss_ab", self.bob_losses[-1], prog_bar=True)
        if current_epoch >= warmup_epochs:
            self.log("loss_eve", self.eve_losses[-1])

    def configure_optimizers(self):
        opt_ab = torch.optim.Adam(list(self.alice.parameters()) + list(self.bob.parameters()) + list(self.keygen.parameters()), lr=self.cfg.lr)
        opt_eves = [torch.optim.Adam(e.parameters(), lr=self.cfg.lr) for e in self.eves]
        return [opt_ab] + opt_eves

    def save_loss_logs(self, out_dir="logs/"):
        import os
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "bob_losses": self.bob_losses,
            "eve_losses": self.eve_losses
        }, os.path.join(out_dir, "loss_log.pt"))

    def on_train_epoch_end(self):
        if self.trainer.current_epoch % self.cfg.log_key_every == 0:
            pub, priv = self.keygen(self.cfg.batch_size)
            self.logger.experiment.add_histogram("pub_key", pub, self.trainer.current_epoch)
            self.logger.experiment.add_histogram("priv_key", priv, self.trainer.current_epoch)

