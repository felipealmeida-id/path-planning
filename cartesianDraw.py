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
data_str = '[[[0, 0], [0, 0], [1, 1], [0, 2], [0, 2], [1, 1], [0, 1], [1, 2], [1, 3], [0, 4], [0, 5], [1, 4], [2, 4], [1, 3], [0, 4], [0, 5], [1, 4], [2, 3], [2, 2], [2, 3], [3, 3], [4, 2], [4, 1], [4, 1], [3, 1], [4, 0], [3, 0], [2, 0], [1, 0], [0, 1], [1, 2], [2, 2], [3, 3], [2, 4], [3, 4], [2, 5], [3, 4], [4, 3], [4, 3], [4, 2], [5, 1], [6, 0], [7, 0], [7, 1], [6, 1], [6, 0], [7, 0], [8, 1], [7, 1], [8, 1], [9, 1], [9, 1], [10, 1], [11, 1], [12, 1], [12, 1], [13, 1], [14, 1], [15, 1], [14, 1], [15, 1], [16, 1], [17, 1], [16, 2], [17, 2], [17, 1], [18, 1], [18, 0], [18, 0], [18, 1], [19, 1], [19, 2], [20, 1], [20, 2], [19, 1], [20, 1], [19, 2], [20, 3], [21, 3], [21, 4], [20, 3], [21, 3], [21, 4], [22, 4], [22, 4], [23, 4], [23, 3], [23, 3], [24, 2], [23, 1], [22, 0], [22, 0], [21, 1], [21, 1], [22, 1], [23, 2], [22, 1], [21, 2], [21, 2], [20, 2], [19, 3], [20, 4], [19, 4], [19, 3], [20, 4], [19, 4], [18, 4], [18, 4], [18, 3], [17, 2], [16, 3], [15, 2], [16, 2], [16, 3], [17, 3], [17, 3], [16, 4], [17, 4], [18, 3], [17, 4], [16, 4], [15, 4], [15, 4], [15, 5], [14, 6], [15, 6], [15, 7], [14, 7], [14, 7], [15, 8], [16, 9], [15, 9], [16, 8], [17, 8], [17, 9], [18, 8], [19, 8], [19, 8], [18, 7], [19, 7], [18, 7], [17, 7], [16, 7], [15, 6], [16, 6], [17, 7], [18, 8], [18, 9], [17, 8], [17, 9], [17, 10], [18, 9], [19, 10], [19, 10], [19, 11], [19, 11], [19, 12], [18, 11], [18, 12], [18, 12], [17, 12], [16, 12], [15, 12], [14, 12], [13, 11], [14, 10], [15, 11], [16, 10], [16, 11], [17, 10], [16, 11], [15, 10], [15, 10], [14, 11], [13, 12], [13, 12], [14, 12], [15, 11], [15, 12], [16, 12], [17, 11], [18, 11], [17, 11], [16, 10], [15, 9], [14, 9], [14, 9], [15, 8], [16, 9], [16, 8], [16, 7], [15, 7], [14, 6], [15, 5], [16, 6], [17, 6], [18, 6], [19, 6], [20, 7], [20, 6], [19, 6], [19, 7], [19, 7], [18, 6], [17, 6], [16, 6], [15, 5], [15, 4], [16, 3], [15, 2], [14, 1], [13, 1], [12, 1], [11, 1], [10, 1], [9, 1], [8, 1], [7, 0], [6, 0], [5, 0], [4, 0], [3, 0], [2, 0], [1, 0], [0, 0]]]'
data = json.loads(data_str)
animate_drones_trayectories_persistent_errors(data)
