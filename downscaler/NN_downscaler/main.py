import os
import torch
from json import load,dump
from torch.nn import Module,Sequential, Linear, BatchNorm1d, Dropout, ReLU, Sigmoid, MSELoss
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
class OHEDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.inputs = []
        self.outputs = []
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
        output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.txt')]
        for input_filepath, output_filepath in zip(input_files, output_files):
            with open(input_filepath, 'r') as f:
                list_input = load(f)
                self.inputs.append(np.array(list_input,dtype=np.float32))
            with open(output_filepath, 'r') as f:
                list_output = load(f)
                self.outputs.append(np.array(list_output,dtype=np.float32))
        self.inputs = torch.tensor(np.array(self.inputs)).to(device)
        self.outputs = torch.tensor(np.array(self.outputs)).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    

class NeuralDownscaler(Module):
    def __init__(self):
        super(NeuralDownscaler, self).__init__()
        self.seq = Sequential(
            Linear(2 * 200 * 4, 2048),
            ReLU(),
            Dropout(0.2),
            Linear(2048, 1024),
            ReLU(),
            Dropout(0.1),
            Linear(1024, 512),
            ReLU(),
            Dropout(0.1),
            Linear(512, 512),
            ReLU(),
            Dropout(0.1),
            Linear(512, 200 * 4),
            Sigmoid()
        )

    def forward(self, x):
        batch_size, num_lines, num_features, num_labels = x.shape
        x = x.view(-1, num_features * num_labels)  # shape becomes [batch_size * num_lines, 400 * num_labels]
        x = self.seq(x)
        x = x.view(batch_size, num_lines, 200, num_labels)
        return x


# Parameters
input_dir = 'encode/input'
output_dir = 'encode/output'
n_epochs = 1000

# Dataset and dataloaders
dataset = OHEDataset(input_dir, output_dir)
val_size = int(0.20 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model, optimizer, and scheduler
model = NeuralDownscaler().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
criterion = MSELoss()

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
        single_input = data[0].round().to(torch.int).cpu().tolist()
        single_output =output[0].detach().round().to(torch.int).cpu().tolist()
        
        with open(f'test/input_epoch_{epoch}.txt','w') as input_file:
            dump(single_input,input_file)
        with open(f'test/output_epoch_{epoch}.txt','w') as output_file:
            dump(single_output,output_file)

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