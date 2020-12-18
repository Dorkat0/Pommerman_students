import time
import numpy as np

from pommerman import agents
from . import group05_utils
from .node import Node
from .mcts import MCTS
from pommerman import constants
from pommerman.forward_model import ForwardModel

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
        self.prev_enemy_pos = None
        self.prev_own_pos = None
        self.rollout_list = None
        self.enemy_value = None
        self.own_value = None
        self.enemy_agent_id = None

        # for forward model:
        self.initialized = False
        self.curr_board = None
        self.curr_agents = None
        self.curr_bombs = None
        self.curr_items = None
        self.curr_flames = None
        self.prev_action = constants.Action.Stop.value
        self.prev_enemy_action = constants.Action.Stop.value

        # save root state
        self.root = None
        self.tree = None

    def get_rollout_list(self):
        return self.rollout_list

    def initialize(self, obs):
        """is called every time a new game starts"""
        if self.agent_id == 0:
            self.own_value = 10
            self.enemy_agent_id = 1
            self.enemy_value = 11
        else:
            self.own_value = 11
            self.enemy_agent_id = 0
            self.enemy_value = 10

        #copied the agent initialization from gamestate.py
        agent_list = list()
        for aid in [10, 11]:
            locations = np.where(obs["board"] == aid)
            agt = agents.DummyAgent()
            agt.init_agent(aid, constants.GameType.FFA)
            if len(locations[0]) != 0:
                agt.set_start_position((locations[0][0], locations[1][0]))
            else:
                agt.set_start_position((0, 0))
                agt.is_alive = False
            agt.reset(is_alive=agt.is_alive)
            agt.agent_id = aid - 10
            agent_list.append(agt)

        self.curr_board, self.curr_agents, self.curr_bombs, self.curr_items, self.curr_flames = ForwardModel.step(
            actions=(constants.Action.Stop.value, constants.Action.Stop.value),
            curr_board=obs["board"].copy(),
            curr_agents=agent_list,
            curr_bombs=[],
            curr_items={},
            curr_flames=[]
        )

        self.prev_enemy_pos = group05_utils.get_enemy_position(obs)
        self.prev_own_pos = obs["position"]

        #init tree
        self.root = Node(self.get_game_state(), self.agent_id)
        self.tree = MCTS(self.agent_id, self.root.state, rollout_depth=4)  # create tree (action space is always 6)
        self.root.find_children()

        self.rollout_list = list()
        self.initialized = True

    def get_game_state(self):
        return [self.curr_board,
                self.curr_agents,
                self.curr_bombs,
                self.curr_items,
                self.curr_flames]

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

        if not self.initialized:
            self.initialize(obs)

        start_time = time.time()
        enemy_position = group05_utils.get_enemy_position(obs)
        self.prev_enemy_action = group05_utils.get_prev_action(obs, self.prev_enemy_pos, enemy_position)
        actual_own_previous_action = group05_utils.get_prev_action(obs, self.prev_own_pos, obs["position"]) #this might happen when there is a collision
        if self.prev_action != actual_own_previous_action:
            self.prev_action = actual_own_previous_action

        self.curr_items = group05_utils.convert_items(obs["board"])

        prev_action_pair = (self.prev_action, self.prev_enemy_action)
        if self.agent_id == 1: # if own agent is second agent then reverse the action pair
            prev_action_pair = prev_action_pair[::-1]

        self.curr_board, self.curr_agents, self.curr_bombs, self.curr_items, self.curr_flames = ForwardModel.step(
                prev_action_pair,
                self.curr_board,
                self.curr_agents,
                self.curr_bombs,
                self.curr_items,
                self.curr_flames)
        if not self.curr_agents[self.enemy_agent_id].is_alive: #a check for a bug that was resolved
            print("enemy_is_alive=", self.curr_agents[self.enemy_agent_id].is_alive)
        group05_utils.boards_are_equal(obs["board"], self.curr_board) #also a check for a bug that was resolved


        # use old tree or create new one
        if prev_action_pair in self.root.children.keys():
            print("use previous")
            self.root = self.root.children[prev_action_pair]
        else:
            self.root = Node(self.get_game_state(), self.agent_id)
            self.tree = MCTS(self.agent_id, self.root.state, rollout_depth=4)  # create tree
        self.root.find_children()  # before we rollout the tree we expand the first set of children
        print("build up_time=", time.time() - start_time)


        start_time = time.time()
        # now rollout tree for 450 ms
        while time.time() - start_time < 0.40:
            self.tree.do_rollout(self.root)
        action = self.tree.choose(self.root)
        self.rollout_list.append(self.tree.N)
        print("n_rollout = ", self.tree.N)

        self.prev_enemy_pos = enemy_position
        self.prev_action = action
        self.prev_own_pos = obs["position"]
        return action

    def episode_end(self, reward):
        print("episode_end")
        self.initialized = False
