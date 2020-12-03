import numpy as np


def get_possible_movements(obs):
    possible_movements = [0, 1, 2, 3, 4, 5]
    position = obs['position']
    barriers = [1, 2, 3, 4]

    # already a bomb on the place        #TODO two bombs are not possible, i guess.
    if (3 == obs['board'][position[0]][position[1]]):
        possible_movements.remove(5)

    # TODO check if bomb is blaced but the enemy of over the bomb

    # boarder or barrier
    if obs['position'][0] > 9 or obs['board'][position[0] + 1][position[1]] in barriers:  # boarder or barrier
        possible_movements.remove(1)  # don´t go further up
    elif obs['position'][0] == 0 or obs['board'][position[0] - 1][position[1]] in barriers:
        possible_movements.remove(2)  # don´t go further down
    if obs['position'][1] > 9 or obs['board'][position[0]][position[1] + 1] in barriers:
        possible_movements.remove(4)  # don´t go further right
    elif obs['position'][1] == 0 or obs['board'][position[0]][position[1] - 1] in barriers:
        possible_movements.remove(3)  # don´t go further left

    return possible_movements


def get_enemy_position(obs):
    pos = obs['enemies'][0].value
    pos = np.where(pos == obs['board'])
    return pos[0][0], pos[1][0]
