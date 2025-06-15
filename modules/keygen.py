import torch
import torch.nn as nn

class KeyGen(nn.Module):
    """
    Neural Key Generator for asymmetric encryption.
    Produces a public/private key pair.
    The keys can be learned or randomized per epoch/message.
    """
    def __init__(self, cfg):
        super().__init__()
        self.key_dim = cfg.key_dim

        self.generator = nn.Sequential(
            nn.Linear(1, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, self.key_dim * 2)  # pub + priv
        )

        self.apply(self.init_weights)

    def forward(self, batch_size):
        seed = torch.ones((batch_size, 1))  # dummy seed
        keys = self.generator(seed)
        pub_key, priv_key = keys.chunk(2, dim=1)
        return pub_key, priv_key

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
