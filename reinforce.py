import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical
from collections import defaultdict
import numpy as np
import gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.device = device
        self.action_shape = action_dim
        self.state_shape = state_dim
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim)
                )
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
                )

    def forward(self, x):
        logits = self.action_layer(x)
        values = self.value_layer(x)
        return logits, values

class Policy:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def act(self, inputs, training=False):
        x = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        logits, values = self.model(x)
        dist = Categorical(logits=logits)
        if training:
            return {"distribution": dist,
                    "values": values}
        else:
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return {"actions": actions.data.cpu().numpy(),
                    "log_probs": log_probs.data.cpu().numpy(),
                    "values": values.data.cpu().numpy()}

class EnvRunner:
  """ Reinforcement learning runner in an environment with given policy """
  def __init__(self, env, policy, nsteps,
               transforms=None, step_var=None):
    self.env = env
    self.policy = policy
    self.nsteps = nsteps
    self.transforms = transforms or []
    self.step_var = step_var if step_var is not None else 0
    self.state = {"latest_observation": self.env.reset()}

  @property
  def nenvs(self):
    """ Returns number of batched envs or `None` if env is not batched """
    return getattr(self.env.unwrapped, "nenvs", None)

  def reset(self):
    """ Resets env and runner states. """
    self.state["latest_observation"] = self.env.reset()
    self.policy.reset()

  def get_next(self):
    """ Runs the agent in the environment.  """
    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []
    self.state["env_steps"] = self.nsteps

    for i in range(self.nsteps):
      observations.append(self.state["latest_observation"])
      act = self.policy.act(self.state["latest_observation"])
      if "actions" not in act:
        raise ValueError("result of policy.act must contain 'actions' "
                         f"but has keys {list(act.keys())}")
      for key, val in act.items():
        trajectory[key].append(val)

      obs, rew, done, _ = self.env.step(trajectory["actions"][-1])
      self.state["latest_observation"] = obs
      rewards.append(rew)
      resets.append(done)
      self.step_var += self.nenvs or 1

      # Only reset if the env is not batched. Batched envs should auto-reset.
      if not self.nenvs and np.all(done):
        self.state["env_steps"] = i + 1
        self.state["latest_observation"] = self.env.reset()

    trajectory.update(observations=observations, rewards=rewards, resets=resets)
    trajectory["state"] = self.state

    for transform in self.transforms:
        transform(trajectory)
    return trajectory

class AsArray:
    """Converts lists of interactions to ndarray."""
    def __call__(self, trajectory):
        # Modify trajectory inplace.
        for k, v in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[k] = np.asarray(v)

class REINFORCE:
    def __init__(self, policy, optimizer, cliprange=0.2, value_loss_coef=0.25,
                 beta = 0.1, max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.beta = beta
        self.max_grad_norm = max_grad_norm

    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory."""
        dist = act['distribution']
        actions = torch.tensor(trajectory['actions'])
        log_probs = dist.log_prob(actions)
        advantages = torch.FloatTensor(trajectory['advantages'])
        L = log_probs*advantages
        return L.mean()

    def entropy_loss(self, act):
        dist = act['distribution']
        entropy_reg = dist.entropy()
        return entropy_reg.mean()

    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        entropy_loss = self.entropy_loss(act)
        loss = -policy_loss - self.beta * entropy_loss
        return loss, entropy_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step."""
        loss, info_data = self.loss(trajectory)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, info_data

def evaluate(policy, env, n_games=1, show=False):
    """Plays an entire game start to end, returns session rewards."""
    game_rewards = []
    for _ in range(n_games):
        # initial observation
        observation = env.reset()
        total_reward = 0
        while True:
            if show:
                env.render()
            actions = policy.act(observation)['actions']
            observation, reward, done, info = env.step(actions)
            total_reward += reward
            if done:
                if show:
                    env.close()
                break
        game_rewards.append(total_reward)
    return game_rewards

class CumReward:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def get_cumulative_rewards(self, rewards, dones, gamma=0.99):
        cum_rewards = np.zeros(len(rewards))
        for i, (r, d) in enumerate(zip(rewards[::-1], dones[::-1])):
            cum_rewards[i] = r + cum_rewards[i-1]*gamma*(1-d)
        return np.asarray(list(reversed(cum_rewards)))

    def __call__(self, trajectory):
        advantages = []
        gae_adv = 0
        val_last = self.policy.act(trajectory['state']['latest_observation'])['values']
        rewards = trajectory['rewards']
        dones = trajectory['resets']
        for i, (reward, done) in enumerate(zip(rewards[::-1], dones[::-1])):
            if done:
                gae_adv = reward
            else:
                gae_adv = self.gamma * gae_adv + reward
            advantages.append(gae_adv)
        trajectory['advantages'] = np.asarray(list(reversed(advantages)))

def train_reinforce(policy, optimizer, env, num_iter=100, upper_limit=float('inf')):
    reinforce = REINFORCE(policy, optimizer)
    for i in range(num_iter):
        runner = EnvRunner(env, policy, nsteps=1000, transforms=[AsArray(), CumReward(policy)])
        reinforce.step(runner.get_next())
        if Summaries.number_of_episodes%10==0:
            reward = np.mean(evaluate(policy, env, n_games=10))
            # print('Reward: {} after {} episodes'.format(reward, Summaries.number_of_episodes))
            if reward > upper_limit:
                print("Env is solved! Reward: ", reward)
                break


if __name__ == '__main__':
    from utils import *

    # env = make_env('CarIntersect-v1', 'reinforce')
    env = make_env('LunarLander-v2', 'reinforce')
    # env = make_env('CartPole-v0', 'reinforce')

    model = Model(env.observation_space.shape[0], env.action_space.n)
    policy = Policy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_reinforce(policy, optimizer, env, num_iter=1000, upper_limit=195)
    evaluate(policy, env, show=True)
