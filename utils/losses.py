import torch.nn.functional as F
import torch
class CompositeLoss:
    def bob_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def eve_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def adversary_loss(self, ciphertext, plaintext, eves):
        return torch.mean(
            torch.stack([F.mse_loss(e(ciphertext), plaintext) for e in eves])
        )
