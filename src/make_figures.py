"""Generate evaluation figures from the saved results JSONs."""
import json

import matplotlib.pyplot as plt
import numpy as np

with open("../reports/bert_results.json") as f:
    bert = json.load(f)
with open("../reports/baseline_results.json") as f:
    baseline = json.load(f)

labels = bert["label_names"]

# --- Confusion matrix (BERT) ---
cm = np.array(bert["confusion_matrix"])
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("BERT fine-tuned — confusion matrix (AG News test set)")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})", ha="center", va="center",
                 color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
fig.colorbar(im, ax=ax, label="Fraction of true class")
fig.tight_layout()
fig.savefig("../reports/figures/confusion_matrix.png", dpi=150)
plt.close(fig)

# --- Per-class F1: baseline vs BERT ---
bert_f1 = [bert["per_class_report"][lbl]["f1-score"] for lbl in labels]
base_f1 = [baseline["per_class_report"][lbl]["f1-score"] for lbl in labels]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width / 2, base_f1, width, label="TF-IDF + LogReg", color="#9ca3af")
ax.bar(x + width / 2, bert_f1, width, label="BERT fine-tuned", color="#2563eb")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1: baseline vs. fine-tuned BERT")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("../reports/figures/f1_comparison.png", dpi=150)
plt.close(fig)

print("Saved figures to ../reports/figures/")
