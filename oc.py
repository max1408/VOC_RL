import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical
from collections import defaultdict, deque, Counter
import gym_car_intersect
import numpy as np
import gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, option_dim, hidden_size=128):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.option_dim = option_dim
        # actor
        self.actor_body = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
                )
        self.action_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim*option_dim)
                )
        self.beta_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, option_dim),
                nn.Sigmoid()
                )
        # critic
        # we calculate only Q_omega instead of Q_U otherwise wasteful:
        self.value_layer_options = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, option_dim)
                )

    def forward(self, x):
        actor = self.actor_body(x)
        policy_logits = self.action_layer(actor).view(-1, self.option_dim, self.action_dim)
        betas = self.beta_layer(actor)
        values = self.value_layer_options(x)
        return policy_logits, betas, values

class Policy:
    def __init__(self, model, fixed_options=None, device="cpu"):
        self.model = model
        self.action_dim = model.action_dim
        self.option_dim = model.option_dim
        self.device = device
        self.batch = None
        self.dim = None
        self.omega = deque(maxlen=2000)
        self.fixed_options = None if fixed_options is None else torch.tensor(fixed_options)

    def options_eps_greedy(self, q_omg, epsilon=0.1):
        # create mask to choose between eps_greedy and argmax strategies:
        size = (1,) if self.dim == 1 else (self.batch, 1)
        mask = (torch.rand(*size) < epsilon).long()
        # make customizable options:
        if self.fixed_options is not None:
            idx_rand = torch.randint(len(self.fixed_options), size)
            idx_max = torch.argmax(q_omg[:,self.fixed_options], dim=-1, keepdim=True)
            idx_max = self.fixed_options[idx_max]
        else:
            idx_rand = torch.randint(self.option_dim, size)
            idx_max = torch.argmax(q_omg, dim=-1, keepdim=True)
        return mask*idx_rand + (1-mask)*idx_max

    def act(self, inputs, options=None, restart=None, epsilon=0.01, training=False):
        """return values corresponding to given options and return new options"""
        x = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        self.batch = 1 if x.dim()==1 else x.shape[0]
        self.dim = x.dim()
        policy_logits, betas, q_omg = self.model(x)
        # epsilon_greedy strategy to sample options:
        if options is None:
            if training:
                raise ValueError("Provide options to train.")
            options = self.options_eps_greedy(q_omg, epsilon)
        else:
            options = torch.tensor(options, dtype=torch.int64).to(self.device)
            # create mask to sample new options if terminates:
            mask = (betas.gather(-1, options) > 0.5).long() # terminates
            # start new options for new episode:
            if restart is not None:
                restart = torch.tensor(restart).long().to(self.device)
                # find idx to restart:
                idx_restart = (restart == 1).nonzero()
                # make sure that restart episode have new options
                mask[idx_restart] = 1
            new_options = self.options_eps_greedy(q_omg, epsilon)
            options = (1-mask)*options + mask*new_options
        # apply options to policies:
        self.omega.append(options.max().item())
        policy_logits_omg = policy_logits[range(self.batch), options.squeeze()]
        # print(f"Full policy shape: {policy_logits.shape}; values {policy_logits}")
        # print(f"Options: {options.shape}; values {options}")
        # print(f"Option policy shape: {policy_logits_omg.shape}; values {policy_logits_omg}")
        dist = Categorical(logits=policy_logits_omg)
        if training:
            return {"distribution": dist,
                    "betas": betas,
                    "options": options,
                    "q_omg": q_omg}
        else:
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            # print('Policy: ', policy_logits_omg)
            # print('Actions: ', actions)
            # print('Beta: ', betas, '\n', options)
            # print('Q_omg: ', q_omg)
            return {"actions": actions.detach().cpu().numpy(),
                    "options": options.detach().cpu().numpy(),
                    "log_prob": log_probs.detach().cpu().numpy(),
                    "betas": betas.detach().cpu().numpy(),
                    "q_omg": q_omg.detach().cpu().numpy()}

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
class EnvOptionPool:
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
        self.options = None
        self.restart = None
        # Whether particular session has just been terminated and needs restarting.
        self.just_ended = [0] * len(self.envs)

    def get_next(self, n_steps=100, epsilon=0.1):
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
                    self.restart[i] = 1
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
            act = self.policy.act(self.prev_observations, self.options, self.restart, epsilon)
            self.options = act['options']
            self.restart = np.zeros_like(self.options)
            actions = act['actions']
            new_obs, cur_rwds, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions))
            # Append data tuple for this tick
            history_log.append((self.prev_observations, actions, self.options, cur_rwds, is_alive))
            self.prev_observations = new_obs
        # cast to numpy arrays, transpose from [time, batch, ...] to [batch, time, ...]
        history_log = list(reversed(history_log)) # reverse time
        # history_log = [np.array(tensor).swapaxes(0, 1) for tensor in zip(*history_log)]
        history_log = [tuple(map(np.asarray, one_pass)) for one_pass in history_log]
        # observation_seq, action_seq, reward_seq, is_alive_seq = history_log
        trajectory = {'history_log': history_log}
        trajectory['last_state'] = np.asarray(self.prev_observations)
        trajectory['last_option'] = self.options
        return trajectory


class OC:
    def __init__(self, policy, optimizer, cliprange=0.2, critic_loss_coef=0.25,
                 entropy_coef = 0.01, beta_reg = 0.01, gamma=0.99):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.beta_reg = beta_reg
        self.gamma = 0.99

    def loss(self, trajectory):
        L_pi = []
        E = []
        L_beta = []
        L_critic = []
        predictions = self.policy.act(trajectory['last_state'], trajectory['last_option'], training=True)
        last_options = torch.LongTensor(trajectory['last_option'])
        betas = predictions['betas'].gather(1, last_options)
        return_util = (1-betas)*predictions['q_omg'].gather(1, last_options) \
                      + betas*torch.max(predictions['q_omg'], dim=1, keepdim=True)[0]
        return_util = return_util.detach()
        iterate = trajectory['history_log']
        for i, (state, action, prev_option, reward, done) in enumerate(iterate):
            reward = torch.FloatTensor(reward)[...,None]
            done = torch.FloatTensor(done)[...,None]
            action = torch.LongTensor(action)
            # calculate return for util function:
            return_util = reward + self.gamma*return_util*(1-done)
            # activate policy:
            act = self.policy.act(state, prev_option, training=True)
            q_omg = act['q_omg']
            option = act['options']
            dist = act['distribution']
            prev_option = torch.LongTensor(prev_option)
            # calculate advantage for policy:
            advantage = return_util - q_omg.gather(1, option)
            # calculate policy loss:
            L_pi.append(dist.log_prob(action) * advantage.squeeze().detach())
            # calculate entropy loss:
            E.append(dist.entropy())
            # calculate critic loss:
            L_critic.append((q_omg.gather(1, option) - return_util.detach())**2)
            # calculate beta loss:
            adv_beta = q_omg.gather(1, prev_option) - torch.max(q_omg, dim=1, keepdim=True)[0] + self.beta_reg
            L_beta.append(act['betas'].gather(1, prev_option)*adv_beta.detach()*(1-done))
        policy_loss = torch.stack(L_pi, dim=1).mean()
        entropy_loss = torch.stack(E, dim=1).mean()
        beta_loss = torch.stack(L_beta, dim=1).mean()
        critic_loss = torch.stack(L_critic, dim=1).mean()
        loss = -policy_loss + beta_loss + self.critic_loss_coef*critic_loss# - self.entropy_coef*entropy_loss
        return loss, (policy_loss, critic_loss, entropy_loss)

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step."""
        loss, info_data = self.loss(trajectory)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, info_data


def evaluate_options(policy, env, n_games=1, show=False):
    """Plays an entire game start to end, returns session rewards."""
    game_rewards = []
    for _ in range(n_games):
        # initial observation
        options = None
        observation = env.reset()
        total_reward = 0
        while True:
            if show:
                env.render()
            act = policy.act([observation], options)
            actions = act['actions'][0]
            options = act['options']
            observation, reward, done, info = env.step(actions)
            total_reward += reward
            if done:
                if show:
                    env.close()
                break
        game_rewards.append(total_reward)
    return game_rewards

def train_oc(policy, optimizer, make_env, num_iter=100, upper_limit=float('inf')):
    epsilon = (max(0.9995**i, 0.01) for i in range(1000000))
    oc = OC(policy, optimizer)
    pool = EnvOptionPool(make_env, policy, n_parallel_games=10)
    for i in range(num_iter):
        eps = next(epsilon)
        oc.step(pool.get_next(10, epsilon=eps))
        if i%100 == 0:
            print(Counter(policy.omega), "epsilon: ", eps)
            reward = np.mean(evaluate_options(policy, env, n_games=4))
            evaluate_options(policy, env, show=True)
            print('Reward: {} after {} episodes'.format(reward, Summaries.number_of_episodes))
            if reward > upper_limit:
                print("Env is solved! Reward: ", reward)
                break

if __name__ == '__main__':
    from utils import *
    from functools import partial

    # env = make_env('CarIntersect-v1', 'oc')
    # func_env = partial(make_env, 'CarIntersect-v1', 'oc')
    env = make_env('LunarLander-v2', 'oc')
    func_env = partial(make_env, 'LunarLander-v2', 'oc')
    # env = make_env('CartPole-v0', 'oc')
    # func_env = partial(make_env, 'CartPole-v0', 'oc')

    model = Model(env.observation_space.shape[0], env.action_space.n, option_dim=2, hidden_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    policy = Policy(model, fixed_options=None)

    train_oc(policy, optimizer, func_env, num_iter=100000, upper_limit=100)

    # evaluate_options(policy, env, show=True)
