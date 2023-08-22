import torch
import torch.nn as nn

# Definición del Discriminador
class Discriminator(nn.Module):
    def __init__(self, hidden_dim=50):
            super(Discriminator, self).__init__()
            
            # LSTM layers
            self.lstm1 = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True, dropout=0.3, num_layers=2)
            # self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, dropout=0.3, num_layers=2)
            
            # Output layer for classification
            self.out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

            #initialize the linear layer from the out layer using he_uniform initialization
            nn.init.kaiming_uniform_(self.out[0].weight, nonlinearity='relu')
            
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        x = self.out(x[:, -1])
        return x