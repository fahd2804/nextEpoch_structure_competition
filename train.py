from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_dataloaders(batch_size = 8, max_length=70, split=0.8, max_data=1000)

model = RNA_net(embedding_dim=64).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300])).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
valid_losses = []
f1s_train = []
f1s_valid = []
for epoch in range(20):

    loss_train = 0.0
    f1_train = 0.0
    loss_valid = 0.0
    f1_valid = 0.0

    for batch in train_loader:

        x = batch["sequence"].to(device) # (N, L)
        y = batch['structure'].to(device) # (N, L, L)

        y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_train += loss.item()
        f1_train += compute_f1(y_pred, y)

    for batch in val_loader:
        x = batch["sequence"].to(device) # (N, L)
        y = batch['structure'].to(device) # (N, L, L)

        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss_valid += loss.item()
        f1_valid += compute_f1(y_pred, y)

    train_losses.append(loss_train/len(train_loader))
    valid_losses.append(loss_valid/len(val_loader))

    f1s_train.append(f1_train/len(train_loader))
    f1s_valid.append(f1_valid/len(val_loader))

    print(f"Epoch {epoch}, F1 train: {f1s_train[-1]:.2f}, F1 valid: {f1s_valid[-1]:.2f}")
    
# Test loop
structures = []
for sequence in test_loader[1]:
    # Replace with your model prediction !
    structure = (model(sequence.to(device).unsqueeze(0)).squeeze(0)>0.5).type(torch.int) # Has to be shape (L, L) ! 
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')
