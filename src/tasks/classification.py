from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.metrics import accuracy


def build_dataloaders(train_ds, test_ds, batch_size: int, num_workers: int = 2):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    for feats, targets in tqdm(loader, desc="train", leave=False):
        feats = feats.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_acc += accuracy(logits, targets) * targets.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for feats, targets in tqdm(loader, desc="eval", leave=False):
            feats = feats.to(device)
            targets = targets.to(device)
            logits = model(feats)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_acc += accuracy(logits, targets) * targets.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def train_supervised_classifier(
    model,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    optimizer,
) -> tuple[list[dict], float, dict]:
    best_acc = -1.0
    best_state = {}
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = eval_one_epoch(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return history, best_acc, best_state


def train_linear_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[list[dict], float]:
    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = nn.Linear(train_x.shape[1], 50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0.0
        for feats, targets in train_loader:
            feats = feats.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()

        train_loss = epoch_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        with torch.no_grad():
            feats = torch.tensor(test_x, dtype=torch.float32, device=device)
            targets = torch.tensor(test_y, dtype=torch.long, device=device)
            logits = model(feats)
            preds = torch.argmax(logits, dim=1)
            test_acc = (preds == targets).float().mean().item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
    return history, history[-1]["test_acc"]
