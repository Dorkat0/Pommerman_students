import torch
import numpy as np


def featurize_simple(obs, device, board_size):
    #np_featurized = _featurize_simple(obs)
    return featurize_centered_to_own_bot(obs, device, board_size)
    # convert to tensor and send to device
    #return torch.tensor([np_featurized]).to(device)


def _featurize_simple(obs):
    # here we encode the board observations into a structure that can
    # be fed into a convolution neural network
    board_size = len(obs['board'])

    # encoding consists of the following seven input planes:
    # 0. board representation
    # 1. bomb_blast_strength
    # 2. bomb_life
    # 3. bomb_moving_direction
    # 4. flame_life
    # 5. own position
    # 6. enemy position

    # board representation
    board_rep = obs['board'].astype(np.float32)

    # encode representation of bombs and flames
    bomb_blast_strength = obs['bomb_blast_strength'].astype(np.float32)
    bomb_life = obs['bomb_life'].astype(np.float32)
    bomb_moving_direction = obs['bomb_moving_direction'].astype(np.float32)
    flame_life = obs['flame_life'].astype(np.float32)

    # encode position of trainee
    position = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    position[obs['position'][0], obs['position'][1]] = 1.0

    # encode position of enemy
    enemy = obs['enemies'][0]  # we only have to deal with 1 enemy
    enemy_position = np.where(obs['board'] == enemy.value, 1.0, 0.0).astype(np.float32)

    # stack all the input planes
    featurized = np.stack((board_rep, bomb_blast_strength, bomb_life, bomb_moving_direction,
                           flame_life, position, enemy_position), axis=0)
    return featurized


def featurize_centered_to_own_bot(obs, device, board_size):
    np_featurized = _featurize_centered_to_own_bot(obs, board_size)
    # convert to tensor and send to device
    return torch.tensor([np_featurized]).to(device)


def _featurize_centered_to_own_bot(obs, board_size):
    # here we encode the board observations into a structure that can
    # be fed into a convolution neural network

    # encoding consists of the following seven input planes:
    # 0. board representation
    # 1. bomb_blast_strength
    # 2. bomb_life
    # 3. bomb_moving_direction
    # 4. flame_life
    # 5. own position
    # 6. enemy position

    # board representation
    board_rep = obs['board'].copy().astype(np.float32)

    position = obs['position']
    board_rep = np.where((board_rep == 1) | (board_rep == 2) | (board_rep == 3), 0, 1)
    board_rep = _get_centered_board(board_rep, position, board_size)


    # encode representation of bombs and flames
    bomb_blast_strength = _get_centered_board(obs['bomb_blast_strength'], position, board_size)
    bomb_life = _get_centered_board(obs['bomb_life'], position, board_size)
    bomb_moving_direction = _get_centered_board(obs['bomb_moving_direction'], position, board_size)
    flame_life = _get_centered_board(obs['flame_life'], position, board_size)

    # encoding position of trainee not necessary because always centered

    # encode position of enemy
    enemy = obs['enemies'][0]  # we only have to deal with 1 enemy
    enemy_position = _get_centered_board(np.where(obs['board'] == enemy.value, 1.0, 0.0), position, board_size)

    # stack all the input planes
    featurized = np.stack((board_rep, bomb_blast_strength, bomb_life, bomb_moving_direction,
                           flame_life, enemy_position), axis=0)
    return featurized


def _get_centered_board(board, position, board_size):
    board = board.copy().astype(np.float32)
    centered_board = np.zeros((board_size, board_size), dtype=np.float32)
    top_edge = position[0] - (board_size // 2)
    bottom_edge = position[0] + (board_size // 2)
    left_edge = position[1] - (board_size // 2)
    right_edge = position[1] + (board_size // 2)

    top_padding = 0
    left_padding = 0
    bottom_padding = 0
    right_padding = 0
    if top_edge < 0:
        top_padding = -top_edge
    if left_edge < 0:
        left_padding = -left_edge
    if bottom_edge > 10:
        bottom_padding = bottom_edge - 10
    if right_edge > 10:
        right_padding = right_edge - 10

    # drop cols and rows that are not visible
    if bottom_edge < 10:
        board = board[:bottom_edge+1, :]
    if right_edge < 10:
        board = board[:, :right_edge+1]
    if top_edge > 0:
        board = board[top_edge:, :]
    if left_edge > 0:
        board = board[:, left_edge:]

    centered_board[top_padding:board_size-bottom_padding, left_padding:board_size-right_padding] = board

    return centered_board
