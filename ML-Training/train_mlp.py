#!/usr/bin/env python3
# train_mlp.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
from sklearn.metrics import classification_report, confusion_matrix

# 1) Dataset-Klasse
class EmbeddingDataset(Dataset):
    def __init__(self, emb_path, labels_csv, scaler=None, encoder=None, fit_scaler=False, fit_encoder=False):
        exclude_countries = {'SI', 'LU', 'MT'}
        df = pd.read_csv(labels_csv)
        mask = ~df['country'].isin(exclude_countries)
        self.y = df.loc[mask, 'country'].values
        self.X = np.load(emb_path)[mask.values]

        # Label-Encoding
        if fit_encoder:
            self.encoder = LabelEncoder()
            self.y = self.encoder.fit_transform(self.y)
        else:
            self.encoder = encoder
            self.y = self.encoder.transform(self.y)

        # Skalierung
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)

        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2) MLP‑Modell
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 3) Training & Evaluation
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * Xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# 4) Main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessor nur auf Trainingsdaten fitten
    train_ds = EmbeddingDataset(
        emb_path=args.train_emb, labels_csv=args.train_labels,
        fit_scaler=True, fit_encoder=True
    )
    val_ds = EmbeddingDataset(
        emb_path=args.val_emb, labels_csv=args.val_labels,
        scaler=train_ds.scaler, encoder=train_ds.encoder
    )
    test_ds = EmbeddingDataset(
        emb_path=args.test_emb, labels_csv=args.test_labels,
        scaler=train_ds.scaler, encoder=train_ds.encoder
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    input_dim   = train_ds.X.shape[1]
    num_classes = len(train_ds.encoder.classes_)

    model     = MLP(input_dim, args.hidden_dims, num_classes, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_one_epoch(model, val_loader,   criterion, device)
        scheduler.step(val_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d} | "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f} | "
              f"LR={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pt'))
            torch.save({
                'scaler': train_ds.scaler,
                'encoder': train_ds.encoder
            }, os.path.join(args.checkpoint_dir, 'preprocessors.pt'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping nach {args.patience} Epochen ohne Verbesserung.")
                break

    # Test-Evaluation
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt')))
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"\n=== Testset: loss={test_loss:.4f}, acc={test_acc:.4f} ===")
    # Ganz oben, falls noch nicht importiert:
# … in deiner main()-Funktion …
    # Early Stopping Loop endet hier
    # …

    # Test-Evaluation
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt')))
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"\n=== Testset: loss={test_loss:.4f}, acc={test_acc:.4f} ===")

    # --- Ausführlicher Report nach Test-Evaluation ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(yb.tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    # Feste Defaults
    args = SimpleNamespace(
        train_emb="emb_train.npy",
        val_emb="emb_val.npy",
        test_emb="emb_test.npy",
        train_labels="y_train.csv",
        val_labels="y_val.csv",
        test_labels="y_test.csv",
        epochs=40,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-5,
        hidden_dims=[512, 256, 128],
        dropout=0.5,
        patience=10,
        lr_factor=0.5,
        lr_patience=5,
        checkpoint_dir="./checkpoints",
    )
    main(args)
