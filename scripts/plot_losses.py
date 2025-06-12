import torch
import matplotlib.pyplot as plt
import os

log_path = "logs/loss_log.pt"
logs = torch.load(log_path)

steps = list(range(len(logs["bob_losses"])))

plt.plot(steps, logs["bob_losses"], label="Bob Loss (Reconstruction)", color="green")
plt.plot(steps, logs["eve_losses"], label="Eve Loss (Interception)", color="red")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Bob vs Eve Loss Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/loss_curve.png")
plt.show()
