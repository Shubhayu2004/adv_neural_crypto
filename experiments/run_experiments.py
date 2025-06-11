import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from trainers.adversarial_trainer import AdvCryptoModel
from data.binary_seq import BinarySequenceDataset
from torch.utils.data import DataLoader

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    model = AdvCryptoModel(cfg)
    ds = BinarySequenceDataset(length=cfg.seq_len, size=cfg.dataset_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=TensorBoardLogger("tb_logs", name=cfg.name),
        optimizer_freq={i:1 for i in range(cfg.num_eves+1)}
    )
    trainer.fit(model, dl)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
