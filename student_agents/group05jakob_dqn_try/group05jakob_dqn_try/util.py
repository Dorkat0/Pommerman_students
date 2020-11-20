import random

import pommerman
from pommerman.agents import DummyAgent, SimpleAgent
from .random_no_bomb_agent import RandomNoBombAgent

def create_training_env():
    # we need this agent just as a placeholder here, the
    # act method will not be called
    trainee = DummyAgent()
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
    return env, trainee, trainee_id, opponent, opponent_id
