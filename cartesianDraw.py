import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

def is_incorrect_move(point1, point2):
    return abs(point1[0] - point2[0]) > 1 or abs(point1[1] - point2[1]) > 1

def animate_drones_trayectories_persistent_errors(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
            if i < len(drone_data):
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
    
    # ax.set_title("Trayectorias de Drones (Animación)")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend(loc="upper left")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=max(len(drone) for drone in data), interval=0, blit=True, repeat=False)
    plt.show()
# Load data from string (simulating json.load)
data_str = '[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 9], [0, 0], [0, 0], [0, 0], [0, 8], [0, 1], [0, 0], [1, 0], [0, 24], [0, 0], [9, 0], [0, 1], [10, 13], [0, 0], [2, 1], [0, 3], [15, 0], [22, 0], [3, 11], [0, 4], [9, 0], [8, 0], [0, 4], [15, 2], [0, 0], [18, 0], [1, 0], [26, 0], [6, 7], [21, 1], [1, 2], [1, 0], [17, 2], [2, 13], [26, 0], [3, 15], [27, 0], [1, 22], [19, 1], [11, 10], [0, 1], [2, 17], [0, 29], [1, 21], [1, 29], [3, 8], [0, 25], [2, 16], [0, 30], [0, 23], [0, 30], [7, 30], [0, 20], [0, 15], [0, 30], [1, 27], [0, 30], [12, 30], [3, 28], [0, 15], [17, 30], [26, 10], [0, 30], [10, 23], [1, 29], [29, 4], [0, 12], [20, 30], [0, 23], [0, 19], [23, 30], [0, 26], [7, 30], [0, 19], [11, 1], [23, 9], [5, 1], [19, 16], [1, 1], [28, 5], [8, 26], [2, 9], [26, 0], [3, 0], [28, 7], [28, 14], [5, 1], [1, 0], [17, 11], [25, 2], [6, 27], [1, 0], [14, 21], [2, 0], [16, 18], [1, 5], [24, 1], [1, 1], [15, 27], [13, 0], [2, 7], [5, 20], [0, 29], [13, 28], [0, 5], [0, 24], [0, 28], [1, 4], [0, 27], [0, 27], [2, 12], [1, 30], [0, 15], [0, 30], [0, 23], [1, 4], [0, 28], [2, 23], [1, 29], [1, 15], [13, 30], [1, 30], [9, 16], [6, 16], [3, 29], [12, 11], [0, 26], [0, 29], [2, 3], [15, 10], [1, 1], [29, 9], [4, 5], [5, 0], [28, 15], [7, 2], [25, 28], [1, 0], [23, 9], [12, 0], [29, 13], [28, 12], [2, 0], [29, 2], [6, 15], [15, 0], [2, 8], [26, 25], [8, 1], [1, 0], [24, 1], [28, 2], [2, 27], [7, 3], [0, 14], [21, 28], [2, 4], [6, 29], [0, 5], [1, 12], [0, 29], [25, 17], [4, 28], [0, 19], [2, 1], [0, 22], [3, 29], [1, 30], [0, 25], [0, 25], [0, 30], [0, 26], [0, 30], [0, 29], [0, 15], [0, 29], [0, 26], [0, 30], [2, 7], [1, 16], [1, 30], [18, 30], [2, 22], [26, 1], [1, 20], [7, 22], [27, 4], [3, 0], [22, 11], [24, 1], [5, 17], [30, 0], [22, 5], [4, 2], [29, 0], [22, 6], [30, 2], [29, 1], [13, 0], [7, 0], [29, 13], [15, 2], [29, 0], [30, 1], [19, 2], [9, 0], [28, 0], [10, 2], [30, 1], [27, 0], [30, 13], [28, 0], [8, 10], [27, 12], [3, 1], [28, 0], [9, 2], [19, 16], [4, 2], [4, 6], [15, 1], [24, 21], [2, 2], [1, 8], [11, 20], [7, 21], [0, 28], [0, 30], [5, 29], [1, 29], [1, 21], [8, 30], [1, 29], [18, 22], [1, 30], [2, 7], [0, 30], [1, 28], [4, 30], [0, 29], [2, 14], [0, 16], [6, 1], [7, 15], [1, 14], [26, 3], [3, 26], [12, 16], [30, 1], [7, 0], [28, 3], [30, 0], [7, 12], [4, 0], [28, 3], [29, 0], [3, 4], [8, 11], [30, 1], [30, 6], [15, 25], [27, 10], [8, 1], [28, 4], [5, 0], [21, 1], [30, 6], [30, 2], [17, 20], [25, 3], [4, 21], [25, 8], [2, 1], [28, 11], [2, 1], [28, 0], [10, 14], [3, 26], [0, 4], [7, 25], [25, 2], [4, 22], [24, 30], [0, 23], [11, 29], [4, 9], [1, 27], [1, 16], [7, 29], [0, 24], [6, 30], [0, 30], [12, 22], [7, 28], [18, 12], [1, 30], [11, 23], [22, 3], [5, 27], [3, 29], [12, 15], [2, 2], [28, 7], [1, 29], [24, 7], [8, 11], [28, 28], [18, 0], [30, 2], [1, 0], [26, 1], [2, 0], [27, 4], [25, 6], [1, 0], [30, 16], [9, 1], [7, 1], [20, 10], [23, 1], [27, 14], [29, 1], [17, 9], [6, 3], [29, 0], [2, 25], [24, 6], [29, 0], [20, 3], [8, 12], [28, 2], [22, 1], [3, 10], [27, 8], [4, 5], [24, 25], [5, 2], [12, 1], [1, 23], [23, 4], [2, 28], [16, 30], [2, 28], [17, 20], [1, 29], [1, 16], [15, 29], [9, 25], [0, 30], [5, 4], [1, 30], [21, 30], [2, 29], [0, 30], [7, 26], [6, 16], [0, 28], [2, 27], [27, 4], [10, 4], [3, 28], [17, 5], [1, 28], [0, 23], [20, 7], [5, 23], [1, 2], [1, 24], [17, 4], [19, 3], [2, 24], [1, 5], [14, 29], [28, 0], [1, 8], [18, 0], [6, 0], [0, 0], [26, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]'
data = json.loads(data_str)
animate_drones_trayectories_persistent_errors(data)
