
# 🔐 Adversarial Neural Cryptography (Symmetric & Asymmetric)

This repository implements neural cryptographic agents inspired by:

> **Learning to Protect Communications with Adversarial Neural Cryptography**  
> Martín Abadi & David G. Andersen — [arXiv:1610.06918](https://arxiv.org/abs/1610.06918)

---

## 🧠 Key Features

| Feature                | Description                                    |
|------------------------|------------------------------------------------|
| ✅ Symmetric crypto     | Shared-key encryption with adversarial Eve    |
| 🔐 Asymmetric crypto    | Public-key Alice / private-key Bob model      |
| ♻️ Key reuse            | Public key reused across batches (optional)   |
| 📊 TensorBoard logging | Track losses and learned key distributions    |
| 📈 Visualizations       | Plot Bob/Eve loss over time                   |
| 🧪 Demo script          | Encrypt/decrypt a message with trained model  |

---

## 📁 Project Structure

```

adv\_neural\_crypto/
├── data/             # Synthetic binary data
├── modules/          # Alice, Bob, Eve, KeyGen networks
├── trainers/         # Lightning training logic
├── utils/            # Loss functions, logging
├── configs/          # YAML training configs
├── scripts/          # Evaluation, plotting, demo
├── experiments/      # Training runner
├── logs/             # Output loss plots, metrics
└── tb\_logs/          # TensorBoard logs

````

---

## 🛠️ Setup

```bash
python -m venv .venv
source .venv/bin/activate          # or .\.venv\Scripts\activate
pip install torch pytorch-lightning pyyaml matplotlib
````

Optional:

```bash
pip install captum
```

---

## 📄 Config Example: `configs/short.yml`

```yaml
name: short
seq_len: 16
key_dim: 16
hidden: 256
batch_size: 512
epochs: 100
lr: 0.001
num_eves: 1
dataset_size: 10000
key_reuse: true
log_key_every: 10
```

---

## 🚀 Training

```bash
python experiments/run_experiments.py configs/short.yml
```

* Logs saved in `logs/loss_log.pt`
* Key distributions appear in TensorBoard

---

## 📊 Visualization

### Loss Curve:

```bash
python scripts/plot_losses.py
```

➡ Generates `logs/loss_curve.png`

### TensorBoard:

```bash
tensorboard --logdir tb_logs/
```

---

## 🧪 Evaluation

```bash
python scripts/evaluate_model.py configs/short.yml
```

Example output:

```
🔐 Avg Bob Decryption Loss : 0.139
🕵️  Avg Eve Interception Loss: 3.74
```

---

## 🔁 Message Demo

Encrypt/decrypt one plaintext example:

```bash
python scripts/demo_asym.py
```

Outputs plaintext, ciphertext, and decrypted result.

---

## ✅ Current Status

| Module        | Status        | Notes                                   |
| ------------- | ------------- | --------------------------------------- |
| Alice / Bob   | ✅ Implemented | MLPs with key support                   |
| Eve           | ✅ Implemented | Symmetric & asymmetric mode             |
| KeyGen        | ✅ Implemented | Learnable public/private key pairs      |
| Trainer       | ✅ Updated     | Manual optimization, warm-up, key reuse |
| Losses        | ✅ Modularized | Supports both symmetric & asymmetric    |
| Logging       | ✅ Active      | Losses + key histograms in TensorBoard  |
| Visualization | ✅ Working     | Plots saved to `logs/`                  |
| Evaluation    | ✅ Scripted    | CLI output of Bob/Eve performance       |
| Demo          | ✅ Minimal     | Scripted 1-shot encryption & decryption |

---

## 🧩 Next Ideas

* Add **transformer-based Alice/Bob**
* Track **bitwise accuracy** in evaluation
* Train Eve on **unseen test keys**
* Apply to **images or text features**

---

## 📝 License

MIT — open for use, extension, and research.

---


