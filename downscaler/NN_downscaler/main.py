import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TrajectoryDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.inputs = []
        self.outputs = []
        
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
        output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.txt')]

        for input_filepath, output_filepath in zip(input_files, output_files):
            with open(input_filepath, 'r') as f:
                lines = f.readlines()
                input_data = [list(map(int, line.strip().split())) for line in lines]
                # Normalize data to [0, 1]
                self.inputs.append(np.array(input_data, dtype=np.float32) / 8)

            with open(output_filepath, 'r') as f:
                lines = f.readlines()
                output_data = [list(map(int, line.strip().split())) for line in lines]
                # Normalize data to [0, 1]
                self.outputs.append(np.array(output_data, dtype=np.float32) / 8)
        
        # Convert list of numpy arrays to a single numpy array and then to a torch tensor
        self.inputs = torch.tensor(np.array(self.inputs)).to(device)
        self.outputs = torch.tensor(np.array(self.outputs)).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    # Method to denormalize data
    @staticmethod
    def denormalize(data):
        return np.round(data * 8)
    
    
class NeuralDownscaler(nn.Module):
    def __init__(self):
        super(NeuralDownscaler, self).__init__()
        self.fc1 = nn.Linear(400, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 200)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, num_lines, num_features = x.shape  # x.shape should be [batch_size, 2, 400]
        
        # Reshape to pass through the network 
        x = x.view(-1, num_features)  # shape becomes [batch_size * 2, 400]
        
        # Forward pass
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)
        # Use sigmoid activation
        x = torch.sigmoid(x)
        
        # Reshape back to [batch_size, 2, 200]
        x = x.view(batch_size, num_lines, -1)

        return x

    

input_dir = 'dataset/input'
output_dir = 'dataset/output'

# Parameters
input_dir = 'dataset/input'
output_dir = 'dataset/output'
n_epochs = 1000

# Dataset and dataloaders
dataset = TrajectoryDataset(input_dir, output_dir)
val_size = int(0.20 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model, optimizer, and scheduler
model = NeuralDownscaler().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
criterion = nn.MSELoss()

# Create directories if they don't exist
os.makedirs('test', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Training loop with model and test file saving every 50 epochs
loss_values = []
val_loss_values = []
patience = 100
early_stop_counter = 0
best_val_loss = float('inf')


for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    loss_values.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_loss_values.append(val_loss)

    # Save model and test files every 50 epochs
    if epoch % 50 == 0:
        torch.save(model.state_dict(), f'model/model_epoch_{epoch}.pt')
        
        # Save a sample input and output for testing
        single_input = data[0].cpu()
        single_output = output[0].detach().cpu()
        
        input_denormalized = TrajectoryDataset.denormalize(single_input)
        np.savetxt(f'test/input_epoch_{epoch}.txt', input_denormalized, fmt='%d')
        
        output_denormalized = TrajectoryDataset.denormalize(single_output)
        np.savetxt(f'test/output_epoch_{epoch}.txt', output_denormalized, fmt='%.6f')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    print(f'Epoch: {epoch}/{n_epochs} | Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}')

    # if early_stop_counter >= patience:
    #     print(f"Early stopping after {patience} epochs with no improvement.")
    #     break
    
    scheduler.step()

# Save the plot as a PNG file
plt.figure(figsize=(12, 6))
plt.plot(range(1, epoch+1), loss_values, label='Training Loss', color='blue')
plt.plot(range(1, epoch+1), val_loss_values, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Training and Validation Loss over Epochs')
plt.savefig('training_validation_loss.png')
plt.close()  # Close the plot so it won't display