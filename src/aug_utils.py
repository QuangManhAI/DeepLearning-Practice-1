import numpy as np
import torch


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def visualize_mixup(loader, class_names, device, alpha=0.2, n_samples=8,
                    save_path='../outputs/images/mixup_samples.png'):
    import matplotlib.pyplot as plt

    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    # Create mixup
    mixed, y_a, y_b, lam = mixup_data(images[:n_samples], labels[:n_samples], alpha)

    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 5))
    for i in range(n_samples):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'{class_names[labels[i]]}', fontsize=7)
        axes[0, i].axis('off')

        axes[1, i].imshow(mixed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title(f'{class_names[y_a[i]]} ({lam:.2f})\n'
                             f'{class_names[y_b[i]]} ({1-lam:.2f})', fontsize=6)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Mixup', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Mixup visualization saved to {save_path}')
