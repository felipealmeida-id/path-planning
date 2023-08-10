class Downscaler:
    movement_delta_dict:dict[int,tuple[int,int]] = {
        0: (0, 0),
        1: (1, 0),
        2: (1, -1),
        3: (0, -1),
        4: (-1, -1),
        5: (-1, 0),
        6: (-1, 1),
        7: (0, 1),
        8: (1, 1),
    }
    delta_movement_dict:dict[tuple[int,int],int] = {
        (0,0):0,
        (1, 0):1,
        (1, -1):2,
        (0, -1):3,
        (-1, -1):4,
        (-1, 0):5,
        (-1, 1):6,
        (0, 1):7,
        (1, 1):8,
    }

    def downscale_trajectory(self,traj_list: list[list[int]]) -> list[list[int]]:
        reduced_traj_list = []
        for traj_str in traj_list:
            x, y = 0, 0
            reduced_traj = []
            for i in range(0, len(traj_str), 2):
                move1 = self.movement_delta_dict[traj_str[i]]
                move2 = self.movement_delta_dict[traj_str[i + 1]]
                x += move1[0] + move2[0]
                y += move1[1] + move2[1]
                big_move_x = x // 2
                big_move_y = y // 2
                x %= 2
                y %= 2
                key = (big_move_x,big_move_y)
                reduced_traj.append(self.delta_movement_dict.get(key))
            reduced_traj_list.append(reduced_traj)
        return reduced_traj_list
