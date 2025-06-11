import torch
from torch.utils.data import Dataset

class BinarySequenceDataset(Dataset):
    def __init__(self, length=16, size=10000):
        self.length = length
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        plaintext = torch.randint(0, 2, (self.length,), dtype=torch.float32)
        key = torch.randint(0, 2, (self.length,), dtype=torch.float32)
        return plaintext, key
