import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import concurrent.futures

# Configurations
sequence_length = 401  # Number of time steps per UAV trajectory
data_dim = 2  # Dimension of the coordinates (x, y)
hidden_dim = 256  # Hidden dimension for LSTM
latent_dim = 512  # Latent space dimension for the generator
batch_size = 64
epochs = 2000
d_learning_rate = 0.0001
g_learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = './out'  # Ensure this directory exists

# Ensure the save path exists
os.makedirs(save_path, exist_ok=True)

class TrajectoryDataset(Dataset):
    def __init__(self, folder_path, transform=None, max_workers=4):
        self.transform = transform
        self.data = []
        json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
        total_files = len(json_files)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.read_json_file, os.path.join(folder_path, file)): file for file in json_files}
            for future in concurrent.futures.as_completed(futures):
                uav_data = future.result()
                self.data.extend(uav_data)  # Flatten the list
                completed += 1
                self.print_progress(completed, total_files, prefix='Loading Data', length=40)

        if self.transform:
            self.data = [self.transform(traj) for traj in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


    @staticmethod
    def read_json_file(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading file {file_path}: {e}")
            return []  # Return an empty list or handle as appropriate

    @staticmethod
    def print_progress(iteration, total, prefix='', length=50):
        percent = 100 * (iteration / total)
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent:.1f}% Complete', end='\r')
        if iteration == total:  # Print a new line when complete
            print()

# Normalization transform
class NormalizeTransform:
    def __init__(self, feature_range=(-1, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, data):
        flat_data = np.concatenate(data, axis=0)
        self.scaler.fit(flat_data)

    def __call__(self, data):
        normalized_data = self.scaler.transform(data)
        return normalized_data

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, sequence_length, data_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        # Batch normalization should be applied across the hidden_dim
        self.bn = nn.BatchNorm1d(sequence_length)
        self.fc = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        h_seq, _ = self.lstm(z)
        # We batch normalize across the hidden dimension
        # No need to reshape since we are normalizing across the hidden_dim
        h_seq_bn = self.bn(h_seq)
        # The rest of the forward pass remains unchanged
        output = self.fc(h_seq_bn)
        output = torch.tanh(output)
        return output
    
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, sequence_length, data_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(data_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        h_last = h_seq[:, -1]
        h_last = self.dropout(h_last)
        output = torch.sigmoid(self.fc(h_last))
        return output

# Initialize the networks
generator = Generator(latent_dim, hidden_dim, sequence_length, data_dim).to(device)
discriminator = Discriminator(sequence_length, data_dim, hidden_dim).to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=g_learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=d_learning_rate)

# Loss function
criterion = nn.BCELoss()

# Load and normalize dataset
folder_path = '../inputs/cartesianSmall/input'  # Replace with your folder path
dataset = TrajectoryDataset(folder_path, transform=None, max_workers=4)
normalize_transform = NormalizeTransform()
normalize_transform.fit(dataset.data)
dataset.data = [normalize_transform(traj) for traj in dataset.data]
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def make_trajectories_valid(trajectories):
    batch_size, sequence_length, data_dim = trajectories.shape
    # Set initial positions to (0,0) for all trajectories
    initial_positions = torch.zeros(batch_size, 1, data_dim).to(device)
    
    # Calculate the differences of each step
    diffs = trajectories[:, 1:] - trajectories[:, :-1]
    
    # Clamp the differences to the range [-1, 1]
    clamped_diffs = torch.clamp(diffs, -1, 1)
    
    # Cumulatively sum the clamped differences starting from (0,0)
    valid_trajectories = torch.cumsum(clamped_diffs, dim=1)
    valid_trajectories = torch.cat([initial_positions, valid_trajectories], dim=1)
    
    # Ensure all points are within the [0, 14] range
    valid_trajectories = torch.clamp(valid_trajectories, 0, 14)

    valid_trajectories = valid_trajectories.int()

    return valid_trajectories

# Training loop
for epoch in range(epochs):
    dloss = 0
    gloss = 0
    for i, real_trajs in enumerate(dataloader):
        # Discriminator training
        real_trajs = real_trajs.to(device)
        real_labels = torch.ones(real_trajs.size(0), 1, device=device)*0.9
        fake_labels = torch.zeros(real_trajs.size(0), 1, device=device) + 0.1

        discriminator.zero_grad()
        real_loss = criterion(discriminator(real_trajs), real_labels)

        # Generate fake trajectories
        noise = torch.randn(real_trajs.size(0), sequence_length, latent_dim, device=device)
        fake_trajs = generator(noise)

        # Scale the output to the correct range and round to integers
        # Use tensors
        # First, normalize to [0, 1]
        fake_trajs = (fake_trajs + 1) / 2

        # Then scale to [0, 14]
        fake_trajs = fake_trajs * 14

        # ROund and int
        fake_trajs = torch.round(fake_trajs).int()

        # Make the generated trajectories valid
        valid_fake_trajs = make_trajectories_valid(fake_trajs)

        # re normalize before sending it to the discriminator
        # First, descale it back from [0, 14] to [0, 1]
        valid_fake_trajs = valid_fake_trajs / 14
        # Now, scale from [0, 1] to the original feature range
        valid_fake_trajs = valid_fake_trajs * (1 - (-1)) + (-1)

        # Discriminator loss on fake trajectories
        fake_loss = criterion(discriminator(valid_fake_trajs.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()

        # Generator training
        generator.zero_grad()
        g_loss = criterion(discriminator(valid_fake_trajs), real_labels)
        g_loss.backward()
        optimizer_g.step()

        # Logging the losses
        dloss += d_loss.item()
        gloss += g_loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], d_loss: {dloss/(i+1):.4f}, g_loss: {gloss/(i+1):.4f}')

    if((epoch + 1) % 10 == 0):
        with torch.no_grad():
            # Generate two separate trajectories
            test_noise = torch.randn(2, sequence_length, latent_dim, device=device)
            generated_traj = generator(test_noise)  # Shape: (2, 401, 2)

            # Scale the output to the correct range and round to integers
            # Use tensors
            # First, normalize to [0, 1]
            generated_traj = (generated_traj + 1) / 2

            # Then scale to [0, 14]
            generated_traj = generated_traj * 14

            # ROund and int
            generated_traj = torch.round(generated_traj).int()
            

            # Make the generated trajectories valid
            valid_trajectories = make_trajectories_valid(generated_traj)
            # Save as JSON
            output_filename = os.path.join(save_path, f'generated_trajectory_epoch_{epoch+1}.json')
            with open(output_filename, 'w') as json_file:
                json.dump(valid_trajectories.tolist()
                        , json_file)


