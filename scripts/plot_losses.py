import torch
import matplotlib.pyplot as plt
import os

log_path = "logs/loss_log.pt"
logs = torch.load(log_path)

steps = list(range(len(logs["bob_losses"])))

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(steps, logs["bob_losses"], label="Bob Loss (Reconstruction)", color="green")
plt.plot(steps, logs["eve_losses"], label="Eve Loss (Interception)", color="red")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Bob vs Eve Loss Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/loss_curve.png")

# Plot bitwise accuracies
plt.figure(figsize=(10, 5))
plt.plot(steps, logs["bob_accuracies"], label="Bob Bitwise Accuracy", color="blue")
plt.plot(steps, logs["eve_accuracies"], label="Eve Bitwise Accuracy", color="orange")
plt.xlabel("Training Step")
plt.ylabel("Bitwise Accuracy")
plt.title("Bob vs Eve Bitwise Accuracy Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/bitwise_accuracy_curve.png")

plt.show()
