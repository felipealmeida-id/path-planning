import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

def is_incorrect_move(point1, point2):
    return abs(point1[0] - point2[0]) > 1 or abs(point1[1] - point2[1]) > 1

def animate_drones_trayectories_persistent_errors(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Lista de colores que excluye el rojo. Puedes agregar más colores si lo necesitas.
    color_list = ['b', 'g', 'c', 'm', 'y', 'k']
    if len(data) > len(color_list):
        raise ValueError("No hay suficientes colores en la paleta para todos los drones.")
    
    lines = []
    current_positions = []
    error_lines = []
    error_markers = []
    
    for drone_index, drone_data in enumerate(data):
        drone_color = color_list[drone_index]
        line, = ax.plot([], [], drone_color + '-', alpha=0.3)  
        lines.append(line)
        current_position, = ax.plot([], [], drone_color + 'o', label='Drone {}'.format(drone_index+1))
        current_positions.append(current_position)
        error_line, = ax.plot([], [], 'r-')  
        error_lines.append(error_line)
        error_marker, = ax.plot([], [], 'rx', markersize=10)  
        error_markers.append(error_marker)
    
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, verticalalignment='top')
    error_points = [[] for _ in data]
    
    def init():
        for line, current_position, error_line, error_marker in zip(lines, current_positions, error_lines, error_markers):
            line.set_data([], [])
            current_position.set_data([], [])
            error_line.set_data([], [])
            error_marker.set_data([], [])
        time_text.set_text('')
        return lines + [time_text] + current_positions + error_markers + error_lines
    
    def animate(i):
        for drone_data, line, current_position, error_line, error_marker in zip(data, lines, current_positions, error_lines, error_markers):
            x_values = [point[0] for point in drone_data[:i+1]]
            y_values = [point[1] for point in drone_data[:i+1]]
            
            if i > 0 and is_incorrect_move(drone_data[i-1], drone_data[i]):
                error_points[data.index(drone_data)].append((drone_data[i-1], drone_data[i]))
            
            error_x = [point[0] for segment in error_points[data.index(drone_data)] for point in segment]
            error_y = [point[1] for segment in error_points[data.index(drone_data)] for point in segment]
            
            line.set_data(x_values, y_values)
            current_position.set_data(x_values[-1], y_values[-1])
            error_line.set_data(error_x, error_y)
            error_marker.set_data(error_x, error_y)
        time_text.set_text('Tiempo: {}'.format(i))
        return lines + [time_text] + current_positions + error_markers + error_lines

    x_min = min([point[0] for drone in data for point in drone]) - 1
    x_max = max([point[0] for drone in data for point in drone]) + 1
    y_min = min([point[1] for drone in data for point in drone]) - 1
    y_max = max([point[1] for drone in data for point in drone]) + 1
    
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xticks(range(0, 31))
    ax.set_yticks(range(0, 31))
    
    ax.set_title("Trayectorias de Drones (Animación)")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend(loc="upper left")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=max(len(drone) for drone in data)+1, interval=100, blit=True, repeat=False)
    plt.show()

# Load data from string (simulating json.load)
data_str = '[[[0, 0], [1, 1], [0, 0], [1, 1], [2, 0], [1, 1], [0, 1], [1, 2], [0, 1], [0, 0], [1, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [1, 13], [0, 12], [1, 13], [2, 14], [2, 13], [2, 14], [3, 14], [2, 14], [2, 13], [2, 12], [3, 12], [2, 12], [1, 12], [2, 13], [3, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8], [8, 7], [8, 6], [8, 5], [8, 4], [8, 3], [8, 2], [8, 1], [7, 2], [8, 1], [7, 0], [6, 0], [6, 1], [6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [1, 14], [2, 13], [1, 12], [2, 12], [2, 11], [3, 11], [2, 11], [1, 10], [2, 10], [1, 9], [0, 9], [0, 8], [1, 9], [0, 10], [1, 9], [2, 8], [3, 7], [4, 6], [5, 5], [6, 4], [7, 3], [8, 2], [8, 1], [7, 2], [6, 2], [6, 3], [7, 3], [8, 3], [7, 4], [8, 4], [8, 5], [7, 6], [6, 7], [5, 8], [4, 9], [3, 10], [2, 11], [1, 12], [0, 13], [1, 14], [0, 13], [1, 14], [0, 13], [1, 12], [0, 12], [1, 12], [1, 11], [1, 10], [2, 9], [1, 9], [2, 8], [3, 7], [4, 8], [5, 7], [6, 6], [7, 5], [8, 4], [8, 3], [8, 2], [8, 1], [7, 2], [8, 1], [8, 2], [9, 2], [8, 1], [9, 1], [8, 0], [7, 0], [6, 0], [6, 1], [5, 2], [4, 1], [3, 0], [2, 1], [1, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [1, 14], [0, 14], [1, 14], [0, 13], [1, 12], [2, 13], [3, 12], [4, 12], [3, 12], [4, 12], [3, 12], [2, 12], [2, 13], [3, 14], [2, 13], [1, 13], [0, 13], [1, 13], [2, 12], [3, 12], [2, 11], [2, 12], [3, 13], [4, 14], [3, 14], [2, 13], [1, 14], [1, 13], [2, 13], [1, 12], [0, 11], [0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 1], [1, 1], [2, 2], [3, 3], [3, 2], [3, 3], [2, 3], [1, 3], [2, 2], [2, 3], [1, 2], [2, 2], [1, 3], [1, 4], [1, 3], [2, 3], [3, 2], [4, 1], [5, 0], [6, 1], [7, 1], [8, 1], [7, 2], [6, 3], [5, 4], [4, 5], [3, 6], [2, 7], [1, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [1, 14], [1, 13], [0, 14], [1, 13], [1, 12], [1, 11], [1, 12], [2, 13], [1, 13], [0, 14], [0, 13], [1, 13], [0, 13], [0, 14], [1, 13], [2, 12], [3, 11], [4, 10], [5, 9], [6, 8], [7, 7], [8, 6], [8, 5], [8, 4], [8, 3], [8, 2], [8, 1], [7, 2], [7, 1], [8, 0], [9, 1], [8, 1], [7, 1], [6, 2], [7, 2], [6, 3], [5, 4], [4, 5], [3, 6], [2, 7], [1, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [1, 13], [0, 14], [1, 14], [0, 13], [0, 12], [0, 13], [1, 12], [0, 13], [1, 13], [0, 14], [0, 13], [0, 14], [1, 13], [2, 12], [3, 11], [4, 10], [5, 9], [6, 8], [7, 7], [8, 6], [8, 5], [8, 4], [8, 3], [8, 2], [8, 1], [9, 0], [10, 1], [11, 2], [10, 3], [9, 4], [8, 5], [7, 6], [6, 7], [5, 8], [4, 9], [3, 10], [2, 11], [1, 12], [0, 13], [0, 12], [1, 12], [0, 12], [1, 12], [1, 13], [0, 14], [1, 13], [2, 12], [3, 12], [4, 12], [3, 13], [2, 13], [1, 13], [0, 13], [1, 14], [2, 14], [2, 13], [1, 13], [0, 12], [1, 13], [2, 13], [3, 14], [4, 14], [5, 13], [4, 13], [4, 14], [4, 13], [4, 14], [5, 13], [4, 13], [3, 14], [3, 13], [3, 12], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], [8, 8], [8, 7], [8, 6], [8, 5], [8, 4], [8, 3], [8, 2], [8, 1], [8, 0], [8, 1], [7, 1], [8, 2], [9, 3], [10, 2], [9, 2], [8, 3], [7, 4], [6, 5], [5, 6], [4, 7], [3, 8], [2, 9], [1, 10], [0, 11], [0, 12], [0, 13], [1, 12], [2, 13], [1, 12], [0, 11], [0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0], [0, 0], [0, 0]]]'
data = json.loads(data_str)
animate_drones_trayectories_persistent_errors(data)
