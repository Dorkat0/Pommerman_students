"""An example to show how to set up an pommerman game programmatically"""
import pommerman
from pommerman import agents
from student_agents.group05.group05 import group05_agent
from student_agents.heuristic_agent.heuristic_agent import heuristic_agent
from student_agents.learning_agent.learning_agent import learning_agent
from student_agents.very_simple_agent.very_simple_agent import very_simple_agent
from group05jakob_dqn_try import util

def main():
    """Simple function to bootstrap a game."""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents
    agent_list = [
        agents.PlayerAgent(),
        group05_agent.Group05Agent(),
        #heuristic_agent.HeuristicAgent(),
        #learning_agent.LearningAgent(),
        #very_simple_agent.VerySimpleAgent(),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(3):
        state = env.reset()
        #util.put_agent_into_other_side(env, agent_list[0], 1)
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
