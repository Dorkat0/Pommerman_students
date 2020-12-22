import time
import numpy as np

from pommerman import agents
from . import group05_utils
from .node import Node
from .mcts import MCTS
from pommerman import constants
from pommerman.forward_model import ForwardModel
from pommerman.constants import Action
from pommerman.constants import Item

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

        # connected sets
        self.reachable_sets = None
        self.reachable_fringes = None
        self.connection_to_enemy = None

    def is_connected_to_enemy(self):
        return self.connection_to_enemy

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

        self.connection_to_enemy = False
        self.reachable_sets = [set(), set()]
        self.reachable_fringes = [list(), list()]
        #expand 2 sets from the starting position
        self.expand((1, 1), 0)
        self.expand((9, 9), 1)

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


        # update reachable sets and firnge and connection to enemy
        if not self.connection_to_enemy:
            for flame in self.curr_flames:
                if flame.life == 0:
                    for i in [0, 1]:
                        for pos in self.reachable_fringes[i]:
                            self.expand(pos, i)
                    break
        if self.connection_to_enemy:
            print("1 connected to enemy")

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

        if self.connection_to_enemy:
            print("2 connected to enemy")

        # use old tree or create new one
        if self.connection_to_enemy:
            # killing mode is on
            if prev_action_pair in self.root.children.keys():
                print("killing mode use previous")
                self.root = self.root.children[prev_action_pair]
            else:
                print("killing mode new tree")
                self.root = Node(self.get_game_state(), self.agent_id)
                self.tree = MCTS(self.agent_id, self.root.state, rollout_depth=4)  # create tree
        else:
            # collecting mode is on
            action_pair = [0, 0]
            action_pair[self.agent_id] = prev_action_pair[self.agent_id]
            action_pair = tuple(action_pair)
            if action_pair in self.root.children.keys():
                print("collecting mode use previous")
                # set enemy action to 0 because we don't care about actual enemy move

                self.root = self.root.children[action_pair]
            else:
                print("should never go in here")
                self.root = Node(self.get_game_state(), self.agent_id)
                self.tree = MCTS(self.agent_id, self.root.state, rollout_depth=4)  # create tree

        self.root.find_children()  # before we rollout the tree we expand the first set of children
        #print("build up_time=", time.time() - start_time)


        start_time = time.time()
        # now rollout tree for 450 ms
        while time.time() - start_time < 0.40:
            self.tree.do_rollout(self.root)
        action = self.tree.choose(self.root)
        self.rollout_list.append(self.tree.N)
        #print("n_rollout = ", self.tree.N)
        self.print_stats()

        self.prev_enemy_pos = enemy_position
        self.prev_action = action
        self.prev_own_pos = obs["position"]
        return action

    def episode_end(self, reward):
        print("episode_end")
        self.initialized = False

    def expand(self, position, agent_id):
        if position in self.reachable_sets[0] and position in self.reachable_sets[1]: # this is our end condition, our field are connected now
            self.connection_to_enemy = True
            return
        children = list()
        is_in_fringe = False #add to fringe if there is a wooden wall as neighbor, because the we want to check it each time a bomb exploded
        for action in range(1, 5): #(1)UP, (2)DOWN, (3)LEFT, (4)RIGHT
            row = position[0]
            col = position[1]
            if action == Action.Up.value:
                row -= 1
            elif action == Action.Down.value:
                row += 1
            elif action == Action.Left.value:
                col -= 1
            elif action == Action.Right.value:
                col += 1

            if row < 0 or row >= len(self.curr_board) or col < 0 or col >= len(self.curr_board) or self.curr_board[row, col] == Item.Rigid.value or (row, col) in self.reachable_sets[agent_id]:
                continue
            elif self.curr_board[row, col] == Item.Wood.value:
                is_in_fringe = True
            else:
                children.append((row, col))
        if position in self.reachable_fringes[agent_id]:
            if not is_in_fringe: # this means that the wood was removed so you can also remove it from the fringe
                self.reachable_fringes[agent_id].remove(position)
        else:
            if is_in_fringe:
                self.reachable_fringes[agent_id].append(position)

        self.reachable_sets[agent_id].add(position)
        for child in children:
            self.expand(child, agent_id)

    def print_stats(self):
        path = list()
        self.print_tree(self.root, path)

    def print_tree(self, node, path):
        print(path)
        path.insert(0, "    ")
        for key, child in node.children.items():
            temp_path = path.copy()
            del temp_path[-1]
            temp_path.append(key)
            self.print_tree(child, temp_path)