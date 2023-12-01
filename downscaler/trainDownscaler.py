from torch import load,tensor,device,cuda,no_grad
from torch.nn import Module,Sequential, Linear, MSELoss, RNN, ReLU
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
from random import randint, random
import numpy as np
from torch import save


HR_TOTAL_TIME=401
TOTAL_TIME=201

def generate_training_data(amount):
  return [[randint(-50,50),randint(-50,50)] for _ in range(HR_TOTAL_TIME)]

def generate_training_output(input:list[list[int]]):
  result = []
  if len(input) % 2 != 0:
    result.append([input[0][0],input[0][1]])
  for i in range(0,len(input) - len(input) % 2,2):
    x_coord = int((input[i][0] + input[i+1][0]) / 2)
    y_coord = int((input[i][1] + input[i+1][1]) / 2)
    result.append([x_coord,y_coord])
  return result

def normalize_tensor(tens):
    mean = tens.mean()
    std = tens.std()
    return (tens-mean)/std

class DataProvider(Dataset):
    def __init__(self,dataset_size):
        input = [generate_training_data(dataset_size) for _ in range(dataset_size)]
        output = list(map(generate_training_output,input))
        self.inputs = tensor(np.array(input,dtype=np.float32)).to(device)
        self.outputs = tensor(np.array(output,dtype=np.float32)).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
class NeuralDownscaler(Module):
    def __init__(self):
        super(NeuralDownscaler, self).__init__()
        self.seq = Sequential(
            Linear(2 * HR_TOTAL_TIME,1024),
            ReLU(),
            Linear(1024,2 * TOTAL_TIME)
        )

    def forward(self, x):
        x = x.view(-1, 2 * HR_TOTAL_TIME)
        x = self.seq(x)
        x = x.view(-1,TOTAL_TIME,2)
        return x
    
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using device: {device}")

n_epochs = 3000
dataset_size = 10000

# Dataset and dataloaders
dataset = DataProvider(dataset_size)
val_size = int(0.20 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model, optimizer, and scheduler
model = NeuralDownscaler().to(device)
optimizer = Adam(model.parameters(), lr=0.00005)
criterion = MSELoss()

loss_values = []
val_loss_values = []


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
    with no_grad():
        for data, target in val_loader:
            output = model(data).round()
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_loss_values.append(val_loss)
    print(f'Epoch: {epoch}/{n_epochs} | Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}')

save(model.state_dict(),'trained_model_401_to_201.pth')