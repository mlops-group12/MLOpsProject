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
    preds, targets: 1D tensors on CPU (or will be moved to CPU)
    Returns: matplotlib.figure.Figure
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    cm = confusion_matrix(targets, preds)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    return fig


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
    images = images.detach().cpu()
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i][0], cmap="gray")
        ax.axis("off")

        pred_name = class_names[preds[i]]
        target_name = class_names[targets[i]]

        color = "green" if preds[i] == targets[i] else "red"
        ax.set_title(f"P: {pred_name}\nT: {target_name}", color=color, fontsize=9)

    fig.tight_layout()
    return fig