import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_fashionmnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def get_augmented_transforms():
    return transforms.Compose([
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=0),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def load_fashionmnist(transform, root='../data', train_transform=None):
    train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True,
        transform=train_transform if train_transform else transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
