import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # For plotting loss
        self.bob_losses = []
        self.eve_losses = []

    def forward(self, pt, key):
        ct = self.alice(pt, key)
        pt_hat = self.bob(ct, key)
        return ct, pt_hat

    def training_step(self, batch, batch_idx):
        pt, key = batch
        warmup_epochs = 10
        current_epoch = self.trainer.current_epoch

        # Generate ciphertext and Bob's reconstruction
        ct = self.alice(pt, key)
        pt_hat = self.bob(ct, key)

        if current_epoch < warmup_epochs:
            # === Warm-up: Train Alice & Bob only ===
            opt_ab = self.optimizers()[0]
            true_bob_loss = self.losses.bob_loss(pt_hat, pt)

            opt_ab.zero_grad()
            self.manual_backward(true_bob_loss)
            opt_ab.step()

            self.bob_losses.append(true_bob_loss.item())
            self.eve_losses.append(float('nan'))

        else:
            # === Adversarial Phase ===
            opt_ab, *opt_eves = self.optimizers()

            # True reconstruction loss
            true_bob_loss = self.losses.bob_loss(pt_hat, pt)

            # Adversarial loss: penalize if Eve succeeds
            adversarial_penalty = self.losses.adversary_loss(ct, pt, self.eves)

            # Composite loss for Alice/Bob
            loss_ab = true_bob_loss - adversarial_penalty

            # Train Alice and Bob
            opt_ab.zero_grad()
            self.manual_backward(loss_ab)
            opt_ab.step()

            # Train each Eve
            for i, eve in enumerate(self.eves):
                pt_pred = eve(ct.detach())
                loss_e = self.losses.eve_loss(pt_pred, pt)
                opt = opt_eves[i]
                opt.zero_grad()
                self.manual_backward(loss_e)
                opt.step()

            # Logging pure losses
            self.bob_losses.append(true_bob_loss.item())
            avg_eve_loss = sum(F.mse_loss(e(ct), pt).item() for e in self.eves) / len(self.eves)
            self.eve_losses.append(avg_eve_loss)

        # Log to TensorBoard
        self.log("loss_ab", true_bob_loss, prog_bar=True)
        if current_epoch >= warmup_epochs:
            self.log("loss_eve", self.eve_losses[-1])


    def configure_optimizers(self):
        opt_ab = torch.optim.Adam(list(self.alice.parameters()) + list(self.bob.parameters()), lr=self.cfg.lr)
        opt_eves = [torch.optim.Adam(e.parameters(), lr=self.cfg.lr) for e in self.eves]
        return [opt_ab] + opt_eves

    def save_loss_logs(self, out_dir="logs/"):
        import os
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "bob_losses": self.bob_losses,
            "eve_losses": self.eve_losses
        }, os.path.join(out_dir, "loss_log.pt"))
