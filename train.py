import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return optimizer

def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch_x, batch_y, batch_target in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        y_pred = model(batch_x, batch_y)
        loss = criterion(y_pred.squeeze(1), batch_target)
        loss.backward()
        optimizer.step()

        # add total training loss for current batch to train_loss
        train_loss += loss.item()*batch_x.shape[0]
    return train_loss

def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y, batch_target in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_target = batch_target.to(device)

            y_pred = model(batch_x, batch_y)
            preds.append(y_pred.detach().cpu().numpy())
            targets.append(batch_target.detach().cpu().numpy())
            loss = criterion(y_pred.squeeze(1), batch_target)
            val_loss += loss.item()*batch_x.shape[0]

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return val_loss, preds, targets

def early_stopping(val_loss, min_val_loss, counter, patience, model):
    if val_loss < min_val_loss:
        print(f"Validation loss decreased ({min_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), 'best_model3.pt')
        min_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter} out of {patience}")
        print(f"Validation loss did not decrease: {val_loss:.6f}")
        if counter >= patience:
            print("Early stopping")
            return True, min_val_loss, counter
    return False, min_val_loss, counter

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=200, patience=50):
    min_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader)
        print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}")

        val_loss, preds, targets = validate(model, criterion, val_loader)

        stop_early, min_val_loss, counter = early_stopping(val_loss, min_val_loss, counter, patience, model)
        if stop_early:
            break

def plot_predictions(preds, targets, scaler_target, title="Actual vs Predicted"):
    preds_inv = scaler_target.inverse_transform(preds.reshape(-1, 1))
    targets_inv = scaler_target.inverse_transform(targets.reshape(-1, 1))

    plt.figure(figsize=(14, 5))
    plt.plot(targets_inv, color='blue', label='Actual values')
    plt.plot(preds_inv, color='red', label='Predicted values')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.legend()
    plt.show()