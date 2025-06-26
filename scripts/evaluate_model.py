import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from types import SimpleNamespace
from trainers.adversarial_trainer import AdvCryptoModel
from data.binary_seq import BinarySequenceDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Dynamically import the correct modules based on config

def load_cfg(config_path):
    cfg_dict = yaml.safe_load(open(config_path))
    return SimpleNamespace(**cfg_dict)

def load_model(cfg, checkpoint_root="tb_logs/midterm-transformer"):
    # Find the latest version directory
    versions = [d for d in os.listdir(checkpoint_root) if d.startswith("version_")]
    latest_version = sorted(versions, key=lambda x: int(x.split("_")[1]))[-1]
    print(f"[INFO] Evaluating checkpoint version: {latest_version}")
    ckpt_path = os.path.join(checkpoint_root, latest_version, "checkpoints")
    ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt")]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")
    ckpt_file = sorted(ckpt_files)[-1]
    full_path = os.path.join(ckpt_path, ckpt_file)

    print(f"Loading checkpoint: {full_path}")
    model = AdvCryptoModel.load_from_checkpoint(full_path, cfg=cfg)
    model.eval()
    return model

def evaluate(model, dataloader):
    total_loss_bob = 0
    total_loss_eve = 0
    total_samples = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for pt, key in dataloader:
            pt = pt.to(device)
            key = key.to(device)
            ct = model.alice(pt, key)
            pt_hat = model.bob(ct, key)
            revs = [e(ct) for e in model.eves]
            batch_size = pt.size(0)
            total_loss_bob += F.mse_loss(pt_hat, pt, reduction='sum').item()
            total_loss_eve += sum(F.mse_loss(r, pt, reduction='sum').item() for r in revs) / len(revs)
            total_samples += batch_size

    print(f"\n📊 Evaluation Results:")
    print(f"🔐 Avg Bob Decryption Loss : {total_loss_bob / total_samples:.6f}")
    print(f"🕵️  Avg Eve Interception Loss: {total_loss_eve / total_samples:.6f}")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/short.yml"
    cfg = load_cfg(config_path)

    # Print which architecture is being used
    if getattr(cfg, 'use_transformer', False):
        print("[INFO] Using transformer-based Alice, Bob, and Eve for evaluation.")
    else:
        print("[INFO] Using MLP-based Alice, Bob, and Eve for evaluation.")

    dataset = BinarySequenceDataset(length=cfg.seq_len, size=cfg.dataset_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)

    model = load_model(cfg)
    evaluate(model, dataloader)