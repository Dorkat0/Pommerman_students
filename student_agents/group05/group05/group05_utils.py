import numpy as np
from pommerman import constants


def get_enemy_position(obs):
    pos = obs['enemies'][0].value
    pos = np.where(pos == obs['board'])
    return pos[0][0], pos[1][0]


def get_prev_action(obs, old_pos, curr_pos):
    if old_pos == curr_pos: #either stop or bomb
        if obs["bomb_life"][curr_pos] == 9:
            return constants.Action.Bomb.value
        else:
            return constants.Action.Stop.value
    elif curr_pos[0] < old_pos[0]:
        return constants.Action.Up.value
    elif curr_pos[0] > old_pos[0]:
        return constants.Action.Down.value
    elif curr_pos[1] < old_pos[1]:
        return constants.Action.Left.value
    elif curr_pos[1] > old_pos[1]:
        return constants.Action.Right.value
    assert False, "This code should not be executed, because it has to be any of the 6 movements: old_enemy_pos=" + str(old_enemy_pos) + " curr_enemy_pos=" + str(curr_enemy_pos)
    return 0


def bomb_can_destroy_a_wooden_wall(board, position, blast_strength):
    return False


def boards_are_equal(board1, board2):
    for r in range(board1.shape[0]):
        for c in range(board1.shape[1]):
            if board1[r][c] != board2[r][c]:
                print("at pos=", r, c, " board1=", board1[r][c], " board2=", board2[r][c])


def convert_items(board):
    """ converts all visible items to a dictionary """
    ret = {}
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            v = board[r][c]
            if v in [constants.Item.ExtraBomb.value,
                     constants.Item.IncrRange.value,
                     constants.Item.Kick.value]:
                ret[(r, c)] = v
    return ret

def avoid_bombs(board):
    # TODO implement
    pass
