import torch, yaml
from types import SimpleNamespace
from modules.keygen import KeyGen
from modules.alice import Alice
from modules.bob import Bob

def load_cfg(path): cfg = yaml.safe_load(open(path)); return SimpleNamespace(**cfg)

def main():
    cfg = load_cfg("configs/short.yml")
    keygen = KeyGen(cfg)
    alice = Alice(cfg); bob = Bob(cfg)

    pub, priv = keygen(1)
    plaintext = torch.randint(0, 2, (1, cfg.seq_len)).float()
    ciphertext = alice(plaintext, pub)
    decrypted = bob(ciphertext, priv)

    print("Plaintext     :", plaintext.numpy())
    print("Ciphertext    :", ciphertext.detach().numpy())
    print("Decrypted     :", decrypted.detach().round().numpy())

if __name__ == "__main__":
    main()
