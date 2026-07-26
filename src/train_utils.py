import torch
from src.aug_utils import mixup_data, mixup_criterion


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, alpha=0.2):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images, y_a, y_b, lam = mixup_data(images, labels, alpha)
        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def train_model(model, train_loader, criterion, optimizer, device,
                num_epochs=10, scheduler=None, use_mixup=False, alpha=0.2, verbose=True):
    train_losses = []
    for epoch in range(num_epochs):
        if use_mixup:
            epoch_loss = train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, alpha)
        else:
            epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(epoch_loss)
        if scheduler:
            scheduler.step()
        if verbose:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {lr:.2e}')
    return train_losses
