import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def explore_dataset(train_dataset, class_names, save_path='../outputs/images/sample_fashionmnist.png'):
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")

    fig = plt.figure(figsize=(14, 10))

    for i in range(10):
        ax = plt.subplot(4, 5, i + 1)
        img, label = train_dataset[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(class_names[label], fontsize=9)
        ax.axis('off')

    ax2 = plt.subplot(4, 5, (11, 15))
    all_labels = train_dataset.targets
    class_counts = [(all_labels == c).sum().item() for c in range(len(class_names))]
    bars = ax2.bar(class_names, class_counts, color='skyblue', edgecolor='navy')
    ax2.set_title('Class Distribution (Train Set)', fontsize=10)
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.set_ylim(top=max(class_counts) * 1.1)
    for bar, count in zip(bars, class_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(count), ha='center', va='bottom', fontsize=7)

    ax3 = plt.subplot(4, 5, (16, 20))
    sample_img = train_dataset[0][0].numpy()
    ax3.hist(sample_img.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7)
    ax3.set_title(f'Pixel Distribution (sample: {class_names[train_dataset[0][1]]})', fontsize=10)
    ax3.set_xlabel('Pixel intensity')
    ax3.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    img_sample, _ = train_dataset[0]
    print(f"Image tensor shape: {img_sample.shape}")
    print(f"Image tensor dtype: {img_sample.dtype}")
    print(f"Pixel value range: [{img_sample.min().item():.4f}, {img_sample.max().item():.4f}]")
    print(f"Mean pixel value: {img_sample.mean().item():.4f}")
    print(f"Std pixel value: {img_sample.std().item():.4f}")


def plot_losses(losses_dict, save_path='../outputs/plots/training_losses.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, losses) in enumerate(losses_dict.items()):
        ax.plot(range(1, len(losses) + 1), losses, marker='o',
                color=colors[idx % len(colors)], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix',
                         save_path='../outputs/plots/confusion_matrix.png'):
    cm_np = cm.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm_np, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=12)

    thresh = cm_np.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm_np[i, j], ha='center', va='center',
                    fontsize=7, color='white' if cm_np[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(probas_dict, true_labels, class_names, n_cols=5,
                    save_path='../outputs/plots/roc_curves.png'):
    n_classes = len(class_names)
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(n_classes):
        ax = axes[i]
        y_true = (true_labels == i).astype(int)
        for idx, (name, probas) in enumerate(probas_dict.items()):
            fpr, tpr, _ = roc_curve(y_true, probas[:, i])
            auc_val = np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                    label=f'{name} (AUC={auc_val:.3f})', linewidth=1.5)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(class_names[i], fontsize=10)
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(alpha=0.3)

    for i in range(n_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pr_curves(probas_dict, true_labels, class_names, n_cols=5,
                   save_path='../outputs/plots/pr_curves.png'):
    n_classes = len(class_names)
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(n_classes):
        ax = axes[i]
        y_true = (true_labels == i).astype(int)
        for idx, (name, probas) in enumerate(probas_dict.items()):
            precision, recall, _ = precision_recall_curve(y_true, probas[:, i])
            ap = np.trapz(precision, recall)
            ax.plot(recall, precision, color=colors[idx % len(colors)],
                    label=f'{name} (AP={ap:.3f})', linewidth=1.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(class_names[i], fontsize=10)
        ax.legend(fontsize=6, loc='lower left')
        ax.grid(alpha=0.3)

    for i in range(n_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_predictions(model, test_loader, class_names, device,
                     save_path='../outputs/plots/predictions.png'):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {class_names[predicted[i]]}\\nTrue: {class_names[labels[i]]}",
                     fontsize=8, color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_misclassified(model, test_loader, class_names, device, max_samples=20,
                       save_path='../outputs/plots/misclassified.png'):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            mask = predicted != labels
            for i in torch.where(mask)[0]:
                misclassified.append((images[i].cpu(), labels[i].cpu(), predicted[i].cpu()))
                if len(misclassified) >= max_samples:
                    break
            if len(misclassified) >= max_samples:
                break

    if not misclassified:
        print('No misclassified samples found.')
        return

    n = len(misclassified)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i in range(n):
        img, true_label, pred_label = misclassified[i]
        ax = axes[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'True: {class_names[true_label]}\\nPred: {class_names[pred_label]}',
                     fontsize=8, color='red')
        ax.axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Displayed {n} misclassified samples.')
