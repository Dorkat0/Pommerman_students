"""
This file shows a basic setup how a reinforcement learning agent
can be trained using DQN. If you are new to DQN, the code will probably be
not sufficient for you to understand the whole algorithm. Check out the
'Literature to get you started' section if you want to have a look at
additional resources.
Note that this basic implementation will not give a well performing agent
after training, but you should at least observe a small increase of reward.
"""

import torch
from torch.nn.functional import mse_loss
import random
import os

from group05jakob_dqn_try import util
from group05jakob_dqn_try.net_input import featurize_simple
from group05jakob_dqn_try.net_architecture import DQN
from group05jakob_dqn_try.replay_memory import ReplayMemory, Transition


def select_action(policy_network, device, obs, eps, n_actions):
    # choose a random action with probability 'eps'
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            # return action with highest q-value (expected reward of an action in a particular state)
            return policy_network(obs).max(1)[1].view(1, 1)
    else:
        # return random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(optimizer, policy_network, target_network, device,
                   replay_memory, batch_size, gamma):
    """
    This function updates the neural network.
    """
    # Sample a batch from the replay memory
    transitions = replay_memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # prepare the batch for further processing
    previous_states = torch.cat(batch.last_state)
    actions = torch.cat(batch.last_action)
    rewards = torch.cat(batch.reward)
    current_states = torch.cat(batch.current_state)
    terminal = torch.cat(batch.terminal)
    non_terminal = torch.tensor(tuple(map(lambda s: not s,
                                          batch.terminal)), device=device, dtype=torch.bool)

    # estimate q-values ( Q(s,a) ) by the policy network
    state_action_values = policy_network(previous_states).gather(1, actions)

    # estimate max_a' Q(s, a') by the target net
    # detach, because we do not need gradients here
    agent_reward_per_action = target_network(current_states).max(1)[0].detach()

    # calculating r + gamma * max_a' Q(s, a'), which serves as target value
    agents_expected_reward = torch.zeros(batch_size, device=device)
    # take only reward if it is a terminal step
    agents_expected_reward[terminal] = rewards[terminal]
    agents_expected_reward[non_terminal] = rewards[non_terminal] + \
        gamma * agent_reward_per_action[non_terminal].squeeze()

    # calculate loss
    loss = mse_loss(state_action_values, agents_expected_reward.unsqueeze(1))

    # set gradients to 0
    optimizer.zero_grad()
    # calculate new gradients
    loss.backward()
    # clip gradients
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    # perform the actual update step
    optimizer.step()


def train(device_name="cuda", model_folder="group05jakob_dqn_try/resources", model_file="model.pt", load_model=False,
          save_model=100, episodes=10000, lr=1e-3, memory_size=100000, min_memory_size=10000, render=False,
          eps_start=1.0, eps_end=0.05, eps_dec=0.00001, batch_size=128, gamma=0.99, print_stats=50, learn_from_agent_for_n_episodes=0):
    device = torch.device(device_name)
    print("Running on device: {}".format(device))

    model_path = os.path.join(model_folder, model_file)

    # create the environment
    env, trainee, trainee_id, opponent, opponent_id, obs_trainee, obs_opponent = util.create_training_env()

    # featurize observations, such that they can be fed to a neural network
    obs_trainee_featurized = featurize_simple(obs_trainee, device)
    obs_size = obs_trainee_featurized.size()

    # create both the policy and the target network
    num_boards = obs_size[1]
    board_size = obs_size[2]
    policy_network = DQN(board_size=board_size, num_boards=num_boards, num_actions=env.action_space.n)
    policy_network.to(device)
    if load_model:
        print("Load model from path: {}".format(model_path))
        policy_network.load_state_dict(torch.load(model_path, map_location=device))
    target_network = DQN(board_size=board_size, num_boards=num_boards, num_actions=env.action_space.n)
    target_network.to(device)
    target_network.load_state_dict(policy_network.state_dict())

    # the optimizer is needed to calculate the gradients and update the network
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
    # DQN is off-policy, it uses a replay memory to store transitions
    replay_memory = ReplayMemory(memory_size)

    episode_count = 0
    reward_count = 0
    # epsilon is needed to control the amount of exploration
    epsilon = eps_start

    reward_list = list()
    # training loop
    while episode_count <= episodes:
        if render:
            env.render()

        # decrease epsilon over time
        if len(replay_memory) > min_memory_size and epsilon > eps_end:
            epsilon -= eps_dec

        # learn from SimpleAgent()
        if learn_from_agent_for_n_episodes > episode_count:
            action = torch.tensor([[trainee.act(obs_trainee, env.action_space)]], device=device, dtype=torch.long)
        else:
            action = select_action(policy_network, device, obs_trainee_featurized, epsilon, env.action_space.n)

        # taking a step in the environment by providing actions of both agents
        actions = [0] * 2
        actions[trainee_id] = action.item()
        # getting action of opponent
        actions[opponent_id] = opponent.act(obs_opponent, env.action_space) #bug fix in here removed .n from opponent.act(obs_opponen, env.action_space.n)
        current_state, reward, terminal, info = env.step(actions)

        obs_trainee_next = current_state[trainee_id]
        obs_trainee_featurized_next = featurize_simple(obs_trainee_next, device)

        ## making my own reward function
        #if not terminal:
        #    reward[trainee_id] += my_reward(obs_trainee_featurized, obs_trainee_featurized_next)

        reward_list.append(reward[trainee_id])
        # preparing transition (s, a, r, s', terminal) to be stored in replay buffer
        reward = float(reward[trainee_id])
        reward = torch.tensor([reward], device=device)
        terminal = torch.tensor([terminal], device=device, dtype=torch.bool)
        replay_memory.push(obs_trainee_featurized, action, reward, obs_trainee_featurized_next, terminal)

        # optimize model if minimum size of replay memory is filled
        if len(replay_memory) > min_memory_size:
            optimize_model(optimizer, policy_network, target_network, device,
                           replay_memory, batch_size, gamma)

        if terminal:
            episode_count += 1
            reward_count += sum(reward_list)  #reward.item()
            reward_list = list()
            if render:
                env.render()
            env.close()

            # create new randomized environment
            env, trainee, trainee_id, opponent, opponent_id, obs_trainee, obs_opponent = util.create_training_env()

            obs_trainee_featurized = featurize_simple(obs_trainee, device)

            if episode_count % save_model == 0:
                torch.save(policy_network.state_dict(), model_path)

            if episode_count % print_stats == 0:
                print("Episode: {}, Reward: {}, Epsilon: {}, Memory Size: {}".format(
                    episode_count, reward_count, epsilon, len(replay_memory)))
                reward_count = 0
        else:
            obs_trainee = obs_trainee_next
            obs_trainee_featurized = obs_trainee_featurized_next
            obs_opponent = current_state[opponent_id]

def my_reward(obs_trainee_featurized:torch.tensor, obs_trainee_featurized_next:torch.tensor):
    board = obs_trainee_featurized[0][0]
    board_next = obs_trainee_featurized_next[0][0]
    board_wooden_walls = (board == 2).sum()
    board_next_wooden_walls = (board_next == 2).sum()
    wooden_walls_removed = board_wooden_walls - board_next_wooden_walls
    reward = wooden_walls_removed * 0.05

    return reward


if __name__ == "__main__":
    model = os.path.join("group05jakob_dqn_try", "resources")
    train(device_name='cpu', model_folder=model, episodes=2000, learn_from_agent_for_n_episodes=0, render=False)
