# visualize.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
    normalize: bool = False,
):
    """
    preds, targets: 1D tensors on CPU
    """
    cm = confusion_matrix(targets, preds)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_example_predictions(
    images: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
    n: int = 8,
):
    """
    images: (N, 1, H, W) tensor on CPU
    preds, targets: 1D tensors on CPU
    """
    plt.figure(figsize=(2 * n, 3))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i][0], cmap="gray")
        plt.axis("off")

        pred_name = class_names[preds[i]]
        target_name = class_names[targets[i]]

        color = "green" if preds[i] == targets[i] else "red"
        plt.title(f"P: {pred_name}\nT: {target_name}", color=color, fontsize=9)

    plt.tight_layout()
    plt.show()
