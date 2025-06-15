import torch
import torch.nn.functional as F

class CompositeLoss:
    def __init__(self):
        pass

    def bob_loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='mean')

    def eve_loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='mean')

    def adversary_loss(self, ciphertext, plaintext, eves):
        """
        For symmetric case: Eve only gets ciphertext.
        """
        return torch.mean(
            torch.stack([F.mse_loss(e(ciphertext), plaintext, reduction='mean') for e in eves])
        )

    def adversary_loss_asymmetric(self, ciphertext, plaintext, pub_key, eves):
        """
        For asymmetric case: Eve gets ciphertext + public key.
        """
        return torch.mean(
            torch.stack([
                F.mse_loss(e(ciphertext, pub_key), plaintext, reduction='mean') for e in eves
            ])
        )
