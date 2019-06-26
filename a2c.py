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

      if not self.nenvs and np.all(done):
        self.state["env_steps"] = i + 1
        self.state["latest_observation"] = self.env.reset()

    trajectory.update(observations=observations, rewards=rewards, resets=resets)
    trajectory["state"] = self.state

    for transform in self.transforms:
        transform(trajectory)
    return trajectory

# A whole lot of space invaders
class EnvPool:
    """
    Sustain several environments simultaniously.
    For each step return nstep future conditions (s_next, r, done, info) (nenv x T)
    Apart from EnvRunner is not provide full trajectory and fit to train by step only.
    """
    def __init__(self, make_env, policy, n_parallel_games=1):
        """
        A special class that handles training on multiple parallel sessions
        and is capable of some auxilary actions like evaluating agent on one
        game session (See .evaluate()).

        :param agent: Agent which interacts with the environment.
        :param make_env: Factory that produces environments OR a name of the
                         gym environment.
        :param n_games: Number of parallel games. One game by default.
        :param epsilon: In case of OptionCritic we need eps-greedy strategy.
                        Pass generator.
        """
        # Create atari games.
        self.policy = policy
        self.make_env = make_env
        self.envs = [self.make_env() for _ in range(n_parallel_games)]
        # Initial observations.
        self.prev_observations = [env.reset() for env in self.envs]
        # Whether particular session has just been terminated and needs restarting.
        self.just_ended = [0] * len(self.envs)

    def get_next(self, n_steps=100):
        """Generate interaction sessions with ataries
        (openAI gym atari environments)
        Sessions will have length n_steps. Each time one of games is finished,
        it is immediately getting reset and this time is recorded in
        is_alive_log (See returned values).

        :param n_steps: Length of an interaction.
        :returns: observation_seq, action_seq, reward_seq, is_alive_seq
        :rtype: a bunch of tensors [batch, tick, ...]
        """
        def env_step(i, action):
            if not self.just_ended[i]:
                new_observation, cur_reward, is_done, info = self.envs[i].step(action)
                if is_done:
                    # Game ends now, will finalize on next tick.
                    self.just_ended[i] = 1
                # note: is_alive=True in any case because environment is still alive
                # (last tick alive) in our notation.
                return new_observation, cur_reward, self.just_ended[i], info
            else:
                # Reset environment, get new observation to be used on next tick.
                new_observation = self.envs[i].reset()
                self.just_ended[i] = 0
                return new_observation, 0, 0, {'end': 1}
        history_log = []
        for i in range(n_steps):
            act = self.policy.act(self.prev_observations)
            actions = act['actions']
            new_obs, cur_rwds, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions))
            # Append data tuple for this tick
            history_log.append((self.prev_observations, actions, cur_rwds, is_alive))
            self.prev_observations = new_obs
        # cast to numpy arrays, transpose from [time, batch, ...] to [batch, time, ...]
        history_log = list(reversed(history_log)) # reverse time
        # history_log = [np.array(tensor).swapaxes(0, 1) for tensor in zip(*history_log)]
        history_log = [tuple(map(np.asarray, one_pass)) for one_pass in history_log]
        # observation_seq, action_seq, reward_seq, is_alive_seq = history_log
        trajectory = {'history_log': history_log}
        trajectory['last_state'] = np.asarray(self.prev_observations)
        return trajectory

class A2C:
    def __init__(self, policy, optimizer, cliprange=0.2, value_loss_coef=0.25,
                 beta = 0.1, gamma=0.99):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.beta = beta
        self.gamma = 0.99

    def loss(self, trajectory):
        L_pi = []
        L_v = []
        E = []
        val_next = self.policy.act(trajectory['last_state'], training=True)['values'].squeeze().detach()
        cum_reward = val_next
        iterate = trajectory['history_log']
        for i, (state, action, reward, done) in enumerate(iterate):
            reward = torch.FloatTensor(reward)
            done = torch.FloatTensor(done)
            action = torch.LongTensor(action)
            act = self.policy.act(state, training=True)
            dist = act['distribution']
            val = act['values'].squeeze()
            # Calculate advantage and target_value
            cum_reward = reward + self.gamma * ((1-done)*cum_reward + done*val_next)
            Adv = cum_reward - val
            L_pi.append(dist.log_prob(action) * Adv.detach())
            L_v.append((reward + self.gamma * val_next - val)**2)
            val_next = val.detach()
            E.append(dist.entropy())
        policy_loss = torch.stack(L_pi, dim=1).mean()
        value_loss = torch.stack(L_v, dim=1).mean()
        entropy_loss = torch.stack(E, dim=1).mean()
        loss = -policy_loss + self.value_loss_coef*value_loss - self.beta*entropy_loss
        return loss, (entropy_loss)

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

def train_a2c(policy, optimizer, make_env, num_iter=100, upper_limit=float('inf')):
    a2c = A2C(policy, optimizer)
    pool = EnvPool(make_env, policy, n_parallel_games=5)
    for i in range(num_iter):
        a2c.step(pool.get_next(100))
        if Summaries.number_of_episodes%100==0:
            reward = np.mean(evaluate(policy, env, n_games=5))
            print('Reward: {} after {} episodes'.format(reward, Summaries.number_of_episodes))
            if reward > upper_limit:
                print("Env is solved! Reward: ", reward)
                break

if __name__ == '__main__':
    from utils import *
    from functools import partial

    # env = make_env('CarIntersect-v1', 'a2c')
    # func_env = partial(make_env, 'CarIntersect-v1', 'a2c')
    # env = make_env('LunarLander-v2', 'a2c')
    # func_env = partial(make_env, 'LunarLander-v2', 'a2c')
    # env = make_env('CartPole-v0', 'a2c')
    # func_env = partial(make_env, 'CartPole-v0', 'a2c')
    env = make_env('Asterix-ram-v0', 'a2c')
    func_env = partial(make_env, 'Asterix-ram-v0', 'a2c')

    model = Model(env.observation_space.shape[0], env.action_space.n, hidden_size=256)
    policy = Policy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_a2c(policy, optimizer, func_env, num_iter=100000, upper_limit=10000)
    evaluate(policy, env, show=True)
