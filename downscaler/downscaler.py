def downscale_trajectory(traj_list: list[list[int]]) -> list[list[int]]:
    # Mapeo de números a movimientos en coordenadas
    movements = {
        "0": (0, 0),
        "1": (1, 0),
        "2": (1, -1),
        "3": (0, -1),
        "4": (-1, -1),
        "5": (-1, 0),
        "6": (-1, 1),
        "7": (0, 1),
        "8": (1, 1),
    }

    reduced_traj_list = []

    for traj_str in traj_list:
        # Coordenadas iniciales del UAV en el cuadrante 2x2
        x, y = 0, 0

        # Resultado de la trayectoria reducida
        reduced_traj = []

        # Procesar la trayectoria original en pares
        for i in range(0, len(traj_str), 2):
            # Obtener los movimientos de los dos pasos
            # print(traj_str[i])
            move1 = movements[str(int(traj_str[i]))]
            move2 = movements[str(int(traj_str[i + 1]))]

            # Aplicar los movimientos dentro del cuadrante 2x2
            x += move1[0] + move2[0]
            y += move1[1] + move2[1]

            # Determinar si hay un movimiento en el mapa de 15x15
            big_move_x = x // 2
            big_move_y = y // 2

            # Restablecer las coordenadas dentro del cuadrante 2x2
            x %= 2
            y %= 2

            # Agregar el movimiento correspondiente a la trayectoria reducida
            for key, value in movements.items():
                if value == (big_move_x, big_move_y):
                    reduced_traj.append(int(key))
                    break

        reduced_traj_list.append(reduced_traj)

    return reduced_traj_list
