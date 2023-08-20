import torch
import torch.nn as nn
import os
import re
import time 


from Generator import Generator
from Discriminator import Discriminator

def denormalize_coordinates(coordinates, max_val=14):
    """Denormaliza una lista de coordenadas del rango de -1 a 1 al rango 0 a max_val y redondea a enteros."""
    return [(int(round(0.5 * (x + 1) * max_val)), int(round(0.5 * (y + 1) * max_val))) for x, y in coordinates]

def normalize_coordinates(coordinates, max_val=14):
    """Normaliza una lista de coordenadas al rango de -1 a 1."""
    return [(2 * (x / max_val) - 1, 2 * (y / max_val) - 1) for x, y in coordinates]


# Definición de la GAN y el proceso de entrenamiento

class GAN():
    def __init__(self, generator: nn.Module, discriminator:nn.Module, latent_dim, real_data, device='cuda'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.real_data = real_data
        self.device = device
        
        # Definimos los optimizadores
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        # Definimos la función de pérdida binaria cruzada
        self.criterion = nn.BCELoss()
    
    def train(self, epochs, batch_size):
        if not os.path.exists('generated_img'):
            os.makedirs('generated_img')
        for epoch in range(epochs):
            start_time = time.time()  # Toma el registro del tiempo al inicio de la época

            # --- Entrenamiento del Discriminador ---
            # self.discriminator.zero_grad()

            # Entrenamiento con datos reales
            real_seqs = self.real_data[:batch_size].to(self.device)
            valid = torch.ones(batch_size, 1).to(self.device)  # Etiquetas para datos reales
            real_loss = self.criterion(self.discriminator(real_seqs), valid)
            
            # Entrenamiento con datos generados
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_seqs = self.generator(z)
            fake = torch.zeros(batch_size, 1).to(self.device)  # Etiquetas para datos generados
            fake_loss = self.criterion(self.discriminator(fake_seqs.detach()), fake)
            
            # Combina las pérdidas y actualiza el Discriminador
            d_loss = (real_loss + fake_loss)
            d_loss.backward()
            self.optimizer_D.step()
            
            # --- Entrenamiento del Generador ---
            for _ in range(2):
                # self.generator.zero_grad()
                
                # Genera datos y trata de engañar al Discriminador
                generated_seqs = self.generator(z)
                g_loss = self.criterion(self.discriminator(generated_seqs), valid)
                
                g_loss.backward()
                self.optimizer_G.step()
            

            # Guardar datos generados cada 10 épocas o en la última época
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                # Generar datos
                z_sample = torch.randn(3, self.latent_dim).to(self.device)
                generated_samples = self.generator(z_sample).detach().cpu().numpy()

                # Denormalizar las coordenadas generadas
                denormalized_samples = [denormalize_coordinates(sample) for sample in generated_samples]
                
                # Guardar en un archivo
                file_path = os.path.join('generated_img', f'epoch_{epoch+1}.txt')
                with open(file_path, 'w') as f:
                    for sample in denormalized_samples:
                        for x, y in sample:
                            f.write(f"({int(round(x))},{int(round(y))})")
            end_time = time.time()  # Toma el registro del tiempo al final de la época
            elapsed_time = end_time - start_time  # Calcula la diferencia de tiempo
            print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] [Time: {elapsed_time:.2f} seconds]")


def load_data_to_tensor_with_normalization(base_path):
    """Carga los datos de las carpetas, los normaliza y los convierte en un tensor de PyTorch."""
    
    def extract_coordinates_from_file(file_path):
        """Extrae las coordenadas (x,y) del archivo y las devuelve como una lista de tuplas."""
        with open(file_path, 'r') as f:
            data = f.read()
        coordinates = re.findall(r'\((\d+),(\d+)\)', data)
        return [(int(x), int(y)) for x, y in coordinates]
    
    # Enumera todos los archivos .txt de las subcarpetas
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()]
    txt_files = []
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        txt_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')])
    
    # Extrae y normaliza las coordenadas de cada archivo
    all_coordinates = [normalize_coordinates(extract_coordinates_from_file(txt_file)) for txt_file in txt_files]
    
    # Convierte las coordenadas normalizadas en un tensor de PyTorch
    data_tensor = torch.FloatTensor(all_coordinates)
    
    return data_tensor

base_path = "input"
normalized_dataset_tensor = load_data_to_tensor_with_normalization(base_path)
normalized_dataset_tensor.shape

# Convertimos los datos normalizados a Tensor de PyTorch
real_data_tensor = torch.FloatTensor(normalized_dataset_tensor)

# Creamos una instancia de la GAN y entrenamos
generator = Generator(latent_dim=100,hidden_dim=400, seq_length=200, )
discriminator = Discriminator(hidden_dim=400)
gan = GAN(generator, discriminator, latent_dim=100, real_data=real_data_tensor, device='cuda')
gan.train(epochs=1000, batch_size=16)