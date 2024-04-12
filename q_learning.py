from logging import Logger
import random
import numpy as np
import tqdm
# define environment
# environment shape:
# XOGOX
# S....
# hidden state: 0,1,2,3,4
# seen state: 0,1
# reward: X = -1, G = 1
# action: left, right, up
logger = Logger()

class Environment:
  def __init__(self) -> None:
    self.hidden_state = 0
    self.terminated = False
    self.truncated = False
    pass
  
  def get_max_state_size(self):
    return 2
  
  def get_max_action_size(self):
    return 3
  
  def reset(self):
    self.hidden_state = 0
    self.terminated = False
    self.truncated = False
  
  def step(self, action):
    if action in self.get_available_actions():
      if action == 'left':
        self.hidden_state -= 1
      elif action == 'right':
        self.hidden_state += 1
      elif action == 'up':
        # TODO
        pass
    else:
      logger.warn(f"invalid action: {action}")
      pass
  
  def get_available_actions(self):
    pass
  
  def get_current_state(self):
    # return seen state
    pass
  
  def get_reward(self):
    pass
  
# define q table
def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable

# define policy
def greedy_policy(Qtable, state):
  # Exploitation: take the action with the highest state, action value
  action = np.argmax(Qtable[state][:])

  return action

def epsilon_greedy_policy(Qtable, state, available_actions, epsilon):
  # Randomly generate a number between 0 and 1
  random_num = random.uniform(0,1)
  # if random_num > greater than epsilon --> exploitation
  if random_num > epsilon:
    # Take the action with the highest value given a state
    # np.argmax can be useful here
    action = greedy_policy(Qtable, state)
  # else --> exploration
  else:
    action = random.choice(available_actions)

  return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
  for episode in tqdm(range(n_training_episodes)):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state, info = env.reset()
    step = 0
    terminated = False
    truncated = False

    # repeat
    for step in range(max_steps):
      # Choose the action At using epsilon greedy policy
      available_actions = env.get_available_actions()
      action = epsilon_greedy_policy(Qtable, state, available_actions, epsilon)

      # Take action At and observe Rt+1 and St+1
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, terminated, truncated, info = env.step(action)
      
      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

      # If terminated or truncated finish the episode
      if terminated or truncated:
        break

      # Our next state is the new state
      state = new_state
  return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param max_steps: Maximum number of steps per episode
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in tqdm(range(n_eval_episodes)):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    step = 0
    truncated = False
    terminated = False
    total_rewards_ep = 0

    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = greedy_policy(Q, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

def start_a_game(Q, env, max_steps):
  total_rewards_ep = 0
  env.
  for step in range(max_steps):
    # Take the action (index) that have the maximum expected future reward given that state
    action = greedy_policy(Q, state)
    new_state, reward, terminated, truncated, info = env.step(action)
    total_rewards_ep += reward
    if terminated or truncated:
      break
    state = new_state
    print(f'current state: {info}')
  return total_rewards_ep

if __name__ == "__main__":
  # create environment
  env = Environment()
  
  state_size = env.get_max_state_size()
  action_size = env.get_max_action_size()
  # define q
  Qtable_frozenlake = initialize_q_table(state_size, action_size)

  # Training parameters
  n_training_episodes = 10000  # Total training episodes
  learning_rate = 0.7          # Learning rate

  # Evaluation parameters
  n_eval_episodes = 100        # Total number of test episodes

  # Environment parameters
  env_id = "Sweeper-v1"     # Name of the environment
  max_steps = 99               # Max steps per episode
  gamma = 0.95                 # Discounting rate
  eval_seed = []               # The evaluation seed of the environment

  # Exploration parameters
  max_epsilon = 1.0             # Exploration probability at start
  min_epsilon = 0.05            # Minimum exploration probability
  decay_rate = 0.0005            # Exponential decay rate for exploration prob

  Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)
  
  mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
  print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
  start_a_game(Qtable_frozenlake, 0, max_steps)