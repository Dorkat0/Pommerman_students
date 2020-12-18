"""An example to show how to set up an pommerman game programmatically"""
import pommerman
from pommerman import agents
from student_agents.group05.group05 import group05_agent
from student_agents.heuristic_agent.heuristic_agent import heuristic_agent
from student_agents.learning_agent.learning_agent import learning_agent
from student_agents.very_simple_agent.very_simple_agent import very_simple_agent
from student_agents.group05jakob_dqn_try.group05jakob_dqn_try import util
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Simple function to bootstrap a game."""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents
    agent_list = [
        #agents.PlayerAgent(),
        #agents.SimpleAgent(),
        agents.SimpleAgent(),
        group05_agent.Group05Agent()
        #heuristic_agent.HeuristicAgent(),
        #learning_agent.LearningAgent(),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    reward_list = list()
    # Run the episodes just like OpenAI Gym
    for i_episode in range(2):
        state = env.reset()
        #util.put_agent_into_other_side(env, agent_list[0], 1)
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished. Reward={}'.format(i_episode, reward))
        reward_list.append(reward)
        plot_list(agent_list[0].get_rollout_list(), "n_rollouts_" + str(i_episode))

    print(reward_list)
    env.close()


def plot_list(y, name):
    print(name)
    print("Average =", np.mean(y))
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(y)
    axes.set_ylim([0, 300])
    axes.set_xlim([0, 600])
    plt.savefig(name)


if __name__ == '__main__':
    main()

