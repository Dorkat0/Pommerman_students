import random
import numpy as np

from pommerman import agents


class Group05Agent(agents.BaseAgent):
    """
    This is the class of your agent. During the tournament an object of this class
    will be created for every game your agents plays.
    If you exceed 500 MB of main memory used, your agent will crash.

    Args:
        ignore the arguments passed to the constructor
        in the constructor you can do initialisation that must be done before a game starts
    """

    def __init__(self, *args, **kwargs):
        super(Group05Agent, self).__init__(*args, **kwargs)
        self.enemy_Items = {6: 1, 7: 2, 8: False}
        self.old_obs = None


    def get_possible_movements(self, obs):
        possible_movements = [0, 1, 2, 3, 4, 5]
        position = obs['position']
        barriers = [1, 2, 3, 4]

        # already a bomb on the place        #TODO two bombs are not possible, i guess.
        if (3 == obs['board'][position[0]][position[1]]):
            possible_movements.remove(5)

        #TODO check if bomb is blaced but the enemy of over the bomb

        # boarder or barrier
        if obs['position'][0] > 9 or obs['board'][position[0] + 1][position[1]] in barriers:     # boarder or barrier
            possible_movements.remove(1)  # don´t go further up
        elif obs['position'][0] == 0 or obs['board'][position[0] - 1][position[1]] in barriers:
            possible_movements.remove(2)  # don´t go further down
        if obs['position'][1] > 9 or obs['board'][position[0]][position[1] + 1] in barriers:
            possible_movements.remove(4)  # don´t go further right
        elif obs['position'][1] == 0 or obs['board'][position[0]][position[1] - 1] in barriers:
            possible_movements.remove(3)  # don´t go further left

        return possible_movements

    def avoid_bombs(self, obs):
        position = obs['position']
        #TODO implement

    def update_enemy_items(self, obs, old_obs):
        if not None:
            pos = obs['enemies'][0].value
            pos = np.where(pos == obs['board'])
            posY = pos[0][0]
            posX = pos[1][0]

            board_status = old_obs['board'][posX][posY]
            if(board_status in (6,7,8)):
                self.enemy_Items[board_status] = self.enemy_Items.get(board_status) + 1

        """
            Returns
            -------
            action: int
                Stop (0): This action is a pass.
                Up (1): Move up on the board.
                Down (2): Move down on the board.
                Left (3): Move left on the board.
                Right (4): Move right on the board.
                Bomb (5): Lay a bomb.
            """

    def act(self, obs, action_space):
        """
        Every time your agent is required to send a move, this method will be called.
        You have 0.5 seconds to return a move, otherwise no move will be played.

        Parameters
        ----------
        obs: dict
            keys:
                'alive': {list:2}, board ids of agents alive
                'board': {ndarray: (11, 11)}, board representation
                'bomb_blast_strength': {ndarray: (11, 11)}, describes range of bombs
                'bomb_life': {ndarray: (11, 11)}, shows ticks until bomb explodes
                'bomb_moving_direction': {ndarray: (11, 11)}, describes moving direction if bomb has been kicked
                'flame_life': {ndarray: (11, 11)}, ticks until flame disappears
                'game_type': {int}, irrelevant for you, we only play FFA version
                'game_env': {str}, irrelevant for you, we only use v0 env
                'position': {tuple: 2}, position of the agent (row, col)
                'blast_strength': {int}, range of own bombs         --|
                'can_kick': {bool}, ability to kick bombs             | -> can be improved by collecting items
                'ammo': {int}, amount of bombs that can be placed   --|
                'teammate': {Item}, irrelevant for you
                'enemies': {list:3}, possible ids of enemies, you only have one enemy in a game!
                'step_count': {int}, if 800 steps were played then game ends in a draw (no points)

        action_space: spaces.Discrete(6)
            action_space.sample() returns a random move (int)
            6 possible actions in pommerman (integers 0-5)
        """

        """
        Returns
        -------
        action: int
            Stop (0): This action is a pass.
            Up (1): Move up on the board.
            Down (2): Move down on the board.
            Left (3): Move left on the board.
            Right (4): Move right on the board.
            Bomb (5): Lay a bomb.
        """


        if self.old_obs is not None:
            self.update_enemy_items(obs, self.old_obs)

        self.old_obs = obs.copy()

        possible = self.get_possible_movements(obs)

        return random.choice(possible)
