import matplotlib.pyplot as plt
import re

# Función para leer el archivo y extraer los datos
def leer_datos_archivo(archivo):
    epochs = []
    evals = []
    g_losses = []
    d_losses = []
    
    try:
        with open(archivo, 'r') as f:
            for linea in f:
                # Mostrar la línea actual que se está procesando
                print(f"Procesando línea: {linea.strip()}")
                # Utilizar regex para extraer los valores
                match = re.match(r'Epoch: (\d+) eval: ([\d\.]+) g_loss: ([\d\.]+) d_loss: ([\d\.]+)', linea)
                if match:
                    epochs.append(int(match.group(1)))
                    evals.append(float(match.group(2)))
                    g_losses.append(float(match.group(3)))
                    d_losses.append(float(match.group(4)))
                else:
                    print(f"Línea no coincide con el patrón esperado: {linea.strip()}")
    except FileNotFoundError:
        print(f"No se pudo encontrar el archivo: {archivo}")
        return [], [], [], []

    print(f"Épocas: {epochs}")
    print(f"Eval: {evals}")
    print(f"G_Loss: {g_losses}")
    print(f"D_Loss: {d_losses}")
    return epochs, evals, g_losses, d_losses

# Leer los datos del archivo
archivo = 'output/finalActions/gan/summary.txt'
epochs, evals, g_losses, d_losses = leer_datos_archivo(archivo)

# Verificar si se han leído datos
if not epochs or not evals or not g_losses or not d_losses:
    print("No se han leído datos. Verifique el archivo de entrada.")
else:
    # Crear la gráfica
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, evals, label='Eval')
    plt.plot(epochs, g_losses, label='G_Loss')
    plt.plot(epochs, d_losses, label='D_Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Valores')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()