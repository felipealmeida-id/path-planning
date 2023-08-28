import torch
import torch.nn as nn

# Modificación del Generador para 200 unidades de tiempo
class Generator(nn.Module):
    def __init__(self, latent_dim, seq_length, hidden_dim=50):
            super(Generator, self).__init__()
            
            self.seq_length = seq_length
            
            # Initial dense layer
            self.dense = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim * self.seq_length),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # LSTM layers
            self.lstm1 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
            
            # Output layer
            self.out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.Tanh()
            )

            # initialize the linear layer from the dense layer using he_uniform initialization
            nn.init.kaiming_uniform_(self.dense[0].weight, nonlinearity='relu')
            
        
    def forward(self, z):
        x = self.dense(z)
        x = x.view(z.size(0), self.seq_length, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.out(x)
        return x