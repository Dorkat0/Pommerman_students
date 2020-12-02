import random

import pommerman
from pommerman.agents import DummyAgent, SimpleAgent
from .random_no_bomb_agent import RandomNoBombAgent

def create_training_env():
    # we need this agent just as a placeholder here, the
    # act method will not be called
    #trainee = DummyAgent()
    ## use SimpleAgent() so our program can learn from it
    trainee = SimpleAgent()
    # as an opponent the SimpleAgent is used
    #opponent = SimpleAgent()
    ## use a DummyAgent for learning because it is easier for the start and it doesn't blow itself up by accident
    opponent = RandomNoBombAgent()
    # we create the ids of the two agents in a randomized fashion
    ids = [0, 1]
    random.shuffle(ids)
    trainee_id = ids[0]
    opponent_id = ids[1]
    agents = [0, 0]
    agents[trainee_id] = trainee
    agents[opponent_id] = opponent
    # create the environment and specify the training agent
    env = pommerman.make('PommeFFACompetition-v0', agents)
    env.set_training_agent(trainee.agent_id)
    env.reset()
    #put_bot_into_other_side()

    put_agent_into_other_side(env, trainee, trainee_id)

    state = env.get_observations()
    obs_trainee = state[trainee_id]
    obs_opponent = state[opponent_id]
    return env, trainee, trainee_id, opponent, opponent_id, obs_trainee, obs_opponent

def put_agent_into_other_side(env, trainee, trainee_id):
    random_x = random.randint(3, 5)
    random_y = random.randint(3, 5)
    while (random_x == 4 and random_y == 4): #avoid placing it on top of the other
        random_x = random.randint(3, 5)
        random_y = random.randint(3, 5)

    if trainee_id == 0:
        env._board[1][1] = 0

        #make space for the unicorn to escape in case it is in the corner
        env._board[9][10] = 0
        env._board[10][9] = 0

        env._board[5+random_x][5+random_y]=10
        trainee.set_start_position((5+random_x, 5+random_y))
        trainee.reset()
    else:
        env._board[9][9]=0
        env._board[0][1] = 0
        env._board[1][0] = 0
        env._board[5-random_x][5-random_y]=11
        trainee.set_start_position((5-random_x, 5-random_y))
        trainee.reset()

