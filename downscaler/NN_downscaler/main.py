import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TrajectoryDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
        self.output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.txt')]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_filepath = self.input_files[idx]
        output_filepath = self.output_files[idx]
        
        with open(input_filepath, 'r') as f:
            lines = f.readlines()
            input_data = [list(map(int, line.strip().split())) for line in lines]
            input_data = np.array(input_data, dtype=np.float32)
        
        with open(output_filepath, 'r') as f:
            lines = f.readlines()
            output_data = [list(map(int, line.strip().split())) for line in lines]
            output_data = np.array(output_data, dtype=np.float32)
        
        return torch.tensor(input_data), torch.tensor(output_data)
    
class NeuralDownscaler(nn.Module):
    def __init__(self):
        super(NeuralDownscaler, self).__init__()
        self.fc1 = nn.Linear(400, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 200)

    def forward(self, x):
        batch_size, num_lines, num_features = x.shape  # x.shape should be [batch_size, 2, 400]
        
        # Reshape to pass through the network
        x = x.view(-1, num_features)  # shape becomes [batch_size * 2, 400]
        
        # Forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape back to [batch_size, 2, 200]
        x = x.view(batch_size, num_lines, -1)
        
        return x

    

input_dir = 'dataset/input'
output_dir = 'dataset/output'

dataset = TrajectoryDataset(input_dir, output_dir)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

model = NeuralDownscaler().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.MSELoss()

n_epochs = 1000  # Increasing the number of epochs to see the saving in action

# Create 'test' folder if it doesn't exist
os.makedirs('test', exist_ok=True)

for epoch in range(1, n_epochs + 1):
    start = time.time()
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target to GPU
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)  # Pass the entire batch at once

        # Compute loss
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    end = time.time()        
    print(f'Epoch: {epoch}/{n_epochs} | Loss: {loss.item():.6f} | Time: {end - start:.2f} s')
    
    # Save model every 50 epochs
    if epoch % 50 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        
        # Save a pair of input and output arrays for checking
        np.savetxt(f'test/input_epoch_{epoch}.txt', data[0].cpu().numpy(), fmt='%d')
        np.savetxt(f'test/output_epoch_{epoch}.txt', output[0].detach().cpu().numpy(), fmt='%.6f')
