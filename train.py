import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline import ChordRecognitionModel
from utils.dataset import ChordDataset
from utils.vocab import ALL_LABELS

# -------- CONFIG --------
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3
HIDDEN_DIM = 128
INPUT_DIM = 252
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- File setup ---------
train_files = [] # TODO: fill with actual list of songs to train with
features_dir = "features/"
labels_dir = "data/labels/"

# -------- Dataset + Loader --------
train_dataset = ChordDataset(features_dir, labels_dir, train_files)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------- Model ---------
model = ChordRecognitionModel(INPUT_DIM, HIDDEN_DIM, num_classes=len(ALL_LABELS))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -------- Training loop --------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")