from enums import Move
all_moves = [Move.DIAG_DOWN_LEFT,Move.DIAG_DOWN_RIGHT,Move.DIAG_UP_LEFT,Move.DIAG_UP_RIGHT,Move.DOWN,Move.LEFT,Move.UP,Move.RIGHT]

nefesto_move_params=["uav","time","uav_index"]
ardemisa_move_params=["uav","time","uav_index","points_of_interest"]
delta_dict = {
    Move.UP:(0,1),
    Move.DIAG_UP_RIGHT:(1,1),
    Move.RIGHT:(1,0),
    Move.DIAG_DOWN_RIGHT:(1,-1),
    Move.DOWN:(0,-1),
    Move.DIAG_DOWN_LEFT:(-1,-1),
    Move.LEFT:(-1,0),
    Move.DIAG_UP_LEFT:(-1,1),
    Move.STAY:(0,0)
}