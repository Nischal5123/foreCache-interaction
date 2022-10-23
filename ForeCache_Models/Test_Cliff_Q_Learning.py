

import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys


if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from gym.envs.toy_text.cliffwalking import CliffWalkingEnv


# matplotlib.style.use('ggplot')


env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
  """
  Creates an epsilon-greedy policy based on a given Q-function and epsilon.

  Args:
      Q: A dictionary that maps from state -> action-values.
          Each value is a numpy array of length nA (see below)
      epsilon: The probability to select a random action. Float between 0 and 1.
      nA: Number of actions in the environment.

  Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.

  """

  def policy_fn(observation):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A

  return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
  """
  Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
  while following an epsilon-greedy policy

  Args:
      env: OpenAI environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Gamma discount factor.
      alpha: TD learning rate.
      epsilon: Chance to sample a random action. Float between 0 and 1.

  Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
  """

  # The final action-value function.
  # A nested dictionary that maps state -> (action -> action-value).
  Q = defaultdict(lambda: np.zeros(env.action_space.n))

  # Keeps track of useful statistics
  # stats = plotting.EpisodeStats(
  #   episode_lengths=np.zeros(num_episodes),
  #   episode_rewards=np.zeros(num_episodes))

  # The policy we're following
  policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

  for i_episode in range(num_episodes):
    # Print out which episode we're on, useful for debugging.
    if (i_episode + 1) % 100 == 0:
      print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
      sys.stdout.flush()

    # Reset the environment and pick the first action
    state = env.reset()

    # One step in the environment
    # total_reward = 0.0
    for t in itertools.count():

      # Take a step
      action_probs = policy(state)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      next_state, reward, done, _ = env.step(action)

      # Update statistics


      # TD Update
      best_next_action = np.argmax(Q[next_state])
      td_target = reward + discount_factor * Q[next_state][best_next_action]
      td_delta = td_target - Q[state][action]
      Q[state][action] += alpha * td_delta

      if done:
        break

      state = next_state

  return Q


Q= q_learning(env, 500)
plotting.plot_episode_stats(stats)



