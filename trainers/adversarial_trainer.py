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

        # 🔁 Key reuse logic
        if self.cfg.key_reuse:
            if not hasattr(self, "_pub_key") or batch_idx == 0:
                self._pub_key, self._priv_key = self.keygen(1)
            pub_key = self._pub_key.expand(batch_size, -1)
            priv_key = self._priv_key.expand(batch_size, -1)
        else:
            pub_key, priv_key = self.keygen(batch_size)

        # 🔐 Compute Bob branch
        ct_bob = self.alice(pt, pub_key)
        pt_hat = self.bob(ct_bob, priv_key)

        # 🟢 Warm-up phase (Bob only)
        if current_epoch < warmup_epochs:
            opt_ab = self.optimizers()[0]
            bob_loss = self.losses.bob_loss(pt_hat, pt)

            opt_ab.zero_grad()
            self.manual_backward(bob_loss)
            opt_ab.step()

            self.bob_losses.append(bob_loss.item())
            self.eve_losses.append(float("nan"))

        else:
            # 🔄 Adversarial phase
            opt_ab = self.optimizers()[0]
            opt_eves = self.optimizers()[1:]

            # Train Bob first
            opt_ab.zero_grad()
            true_bob_loss = self.losses.bob_loss(pt_hat, pt)
            self.manual_backward(true_bob_loss, retain_graph=True)  # Retain for Eve's training
            opt_ab.step()

            # Train Eves
            for i, (eve, opt_eve) in enumerate(zip(self.eves, opt_eves)):
                opt_eve.zero_grad()
                eve_pred = eve(ct_bob)
                eve_loss = self.losses.eve_loss(eve_pred, pt)
                
                # Retain graph for all but the last Eve
                retain = i < len(self.eves) - 1
                self.manual_backward(eve_loss, retain_graph=retain)
                opt_eve.step()

            self.bob_losses.append(true_bob_loss.item())
            self.eve_losses.append(eve_loss.item())

        # 📈 TensorBoard log
        self.log("loss_ab", self.bob_losses[-1], prog_bar=True)
        if current_epoch >= warmup_epochs:
            self.log("loss_eve", self.eve_losses[-1], prog_bar=True)



    def configure_optimizers(self):
        opt_ab = torch.optim.Adam(
            list(self.alice.parameters()) + list(self.bob.parameters()) + list(self.keygen.parameters()),
            lr=self.cfg.lr
        )
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
        if self.cfg.log_key_every and self.trainer.current_epoch % self.cfg.log_key_every == 0:
            pub, priv = self.keygen(self.cfg.batch_size)
            self.logger.experiment.add_histogram("pub_key", pub, self.trainer.current_epoch)
            self.logger.experiment.add_histogram("priv_key", priv, self.trainer.current_epoch)
