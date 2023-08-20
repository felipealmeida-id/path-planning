import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import re
import numpy as np

def is_move_valid(prev, curr):
    # Comprobar si el movimiento es válido (una unidad en cualquier dirección)
    return abs(prev[0] - curr[0]) <= 1 and abs(prev[1] - curr[1]) <= 1

def animate_trajectory(i, trajectory, line, current_point, start_point, passed_points, time_text, invalid_moves):
    trajectory_np = np.array(trajectory)
    x, y = trajectory_np[i]
    
    # Check for invalid moves
    if i > 0:
        prev_x, prev_y = trajectory_np[i-1]
        if abs(x - prev_x) > 1 or abs(y - prev_y) > 1:
            invalid_moves.append((prev_x, prev_y, x, y))
    
    current_point.set_data([x], [y])
    line.set_data(trajectory_np[:i+1, 0], trajectory_np[:i+1, 1])
    
    # Update passed points data
    passed_points.set_xdata(trajectory_np[:i+1, 0])
    passed_points.set_ydata(trajectory_np[:i+1, 1])
    
    time_text.set_text(f"Time: {i}")
    
    for start_x, start_y, end_x, end_y in invalid_moves:
        plt.plot([start_x, end_x], [start_y, end_y], color="red")

    return line, current_point, start_point, passed_points, time_text


def plot_trajectories_from_txt_v2(file_path):
    # Leer el archivo y obtener las trayectorias
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Usar regex para extraer las coordenadas (x, y)
    pattern = re.compile(r'\((-?\d+),\s*(-?\d+)\)')
    
    trajectories = []
    for line in lines:
        matches = pattern.findall(line)
        trajectory = [tuple(map(float, match)) for match in matches]
        centroid_trajectory = [(round(point[0]) + 0.5, round(point[1]) + 0.5) for point in trajectory]
        trajectories.append(centroid_trajectory)
    
    # Verificación
    if not trajectories or not trajectories[0]:
        print(f"El archivo {os.path.basename(file_path)} no tiene trayectorias válidas o está vacío.")
        return
    
    global ax
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_xticks([i for i in range(16)])
    ax.set_yticks([i for i in range(16)])
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.set_title(f"Trayectorias del archivo {os.path.basename(file_path)}")
    
    line, = ax.plot([], [], lw=2, color='blue', alpha=0.5)
    current_point, = ax.plot([], [], marker='o', color='red', markersize=5)
    passed_points, = ax.plot([], [], 'o', color='blue', markersize=3)
    start_point, = ax.plot(trajectories[0][0][0], trajectories[0][0][1], marker='o', color='green', markersize=7)
    time_text = ax.text(0.85, 0.95, '', transform=ax.transAxes)
    invalid_moves = []
    
    ani = FuncAnimation(fig, animate_trajectory, frames=len(trajectories[0]), fargs=(trajectories[0], line, current_point, start_point, passed_points, time_text, invalid_moves), blit=False, interval=100, repeat=False)
    plt.show()

plot_trajectories_from_txt_v2("gen_images/epoch_10_sample_0.txt")

