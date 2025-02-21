
import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]  # Take last time step output
        output = self.fc(final_out)
        return output  # Pass through linear layer

    def train_with_early_stopping(self, train_loader, val_loader, num_epochs, optimizer, criterion, device, patience=10):
        """ Training function with early stopping. """
        self.to(device)
        best_val_loss = float('inf')
        best_val_epoch = -1
        best_train_loss = float('inf')

        epochs_without_improvement = 0
        best_model_state = None
        train_losses = []  # To store the train loss values
        val_losses = []    # To store the validation loss values
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.mean(axis=1, keepdims=True)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)  # Average the loss over the batch

            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = targets.mean(axis=1, keepdims=True)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)  # Average the loss over the batch

            # Append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # # Optionally print progress
            # if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            #     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

            # Check for improvement
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_val_epoch = epoch + 1
                best_train_loss = train_losses[-1]  # Store the train loss at the best val_loss epoch
                best_model_state = self.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model weights
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        # Print the epoch with the lowest validation loss and corresponding training loss
        if best_val_epoch != -1:
            print(f"The lowest validation loss occurred at epoch {best_val_epoch} with a train loss of {best_train_loss:.4f}, val loss of {best_val_loss:.4f}")
        # Return the train and validation losses
        return train_losses, val_losses

