import matplotlib.pyplot as plt

# Function to read the data from the text file
def read_data(filename):
    epochs = []
    g_losses = []
    d_losses = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            epoch = int(parts[1])
            g_loss = float(parts[5])
            d_loss = float(parts[7])
            epochs.append(epoch)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
    
    return epochs, g_losses, d_losses

# Function to plot the data
def plot_data(epochs, g_losses, d_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, g_losses, label='Loss del generador', color='blue')
    plt.plot(epochs, d_losses, label='Loss del discriminador', color='red')
    
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    filename = './output/newCartesian_1/gan/summary.txt'  # Replace with your file name
    epochs, g_losses, d_losses = read_data(filename)
    plot_data(epochs, g_losses, d_losses)

if __name__ == '__main__':
    main()