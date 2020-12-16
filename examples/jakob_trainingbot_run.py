"""An example to show how to set up an pommerman game programmatically"""
import pommerman
from pommerman import agents
from student_agents.group05jakob_dqn_try.group05jakob_dqn_try import learning_agent
from student_agents.group05jakob_dqn_try.group05jakob_dqn_try import random_no_bomb_agent
from student_agents.group05jakob_dqn_try.group05jakob_dqn_try import util

def main():
    """Simple function to bootstrap a game."""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents
    agent_list = [
        random_no_bomb_agent.RandomNoBombAgent(),
        learning_agent.LearningAgent()
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(3):
        state = env.reset()
        util.put_agent_into_other_side(env, agent_list[0], 0)
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished. Reward={}'.format(i_episode, reward))
    env.close()


if __name__ == '__main__':
    main()
