import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from trainers.adversarial_trainer import AdvCryptoModel
from data.binary_seq import BinarySequenceDataset
from torch.utils.data import DataLoader
from types import SimpleNamespace

def main(config_path):
    cfg_dict = yaml.safe_load(open(config_path))
    cfg = SimpleNamespace(**cfg_dict)
    model = AdvCryptoModel(cfg)
    ds = BinarySequenceDataset(length=cfg.seq_len, size=cfg.dataset_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=TensorBoardLogger("tb_logs", name=cfg.name),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1

    )
    trainer.fit(model, dl)
    model.save_loss_logs("logs/")
    print("✅ Training complete. Loss logs saved to logs/loss_log.pt")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
