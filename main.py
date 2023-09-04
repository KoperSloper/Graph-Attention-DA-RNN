from dataset import train_loader, val_loader, test_loader, scaler_target
from model import initialize_model
from train import initialize_optimizer, validate, train_model, plot_predictions
from torch import nn
import torch

# Initialize model
time_series = train_loader.dataset.tensors[0].shape[-2]
hidden_dim1 = 128
hidden_dim2 = 128
timesteps = train_loader.dataset.tensors[0].shape[1]
features = train_loader.dataset.tensors[0].shape[-1]

model = initialize_model(time_series, hidden_dim1, hidden_dim2, timesteps, features)

# Initialize optimizer and scheduler
optimizer = initialize_optimizer(model)

# Train model
train_model(model=model, criterion=nn.MSELoss(), optimizer=optimizer, train_loader=train_loader, val_loader=val_loader)

model.load_state_dict(torch.load('best_model3.pt'))

# Validate model and plot predictions
val_loss, preds_val, targets_val = validate(model=model, criterion=nn.MSELoss(), val_loader=val_loader)
print(f"Validation loss: {val_loss:.6f}")
plot_predictions(preds_val, targets_val, scaler_target, "Validation: Actual vs predicted")

# Test model and plot predictions
test_loss, preds_test, targets_test = validate(model=model, criterion=nn.MSELoss(), val_loader=test_loader)
print(f"Test loss: {test_loss:.6f}")
plot_predictions(preds_test, targets_test, scaler_target, "Test: Actual vs predicted")