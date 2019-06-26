import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict, deque, Counter
import gym_car_intersect
import numpy as np
import gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvModel(nn.Module):
    def __init__(self, state_dim, action_dim, option_dim, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.option_dim = option_dim
        # convolution net:
        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.MaxPool2d(kernel_size=(3, 3)),
                                  nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                                  nn.ELU(),
                                  nn.MaxPool2d(kernel_size=(3, 3)),
                                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                                  nn.ELU(),
                                  nn.MaxPool2d(kernel_size=(3, 3)),
                                  Flatten())
        size = np.prod(self.conv(torch.zeros(1, *self.state_dim)).shape[1:])
        # actor
        self.actor_body = nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                )
        self.action_layer = nn.Sequential(
                nn.Linear(hidden_size, action_dim*option_dim)
                )
        self.beta_layer = nn.Sequential(
                nn.Linear(hidden_size, option_dim),
                nn.Sigmoid()
                )
        # critic
        # we calculate only Q_omega instead of Q_U otherwise wasteful:
        self.value_layer_options = nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, option_dim)
                )

    def forward(self, x):
        x = self.conv(x)
        actor = self.actor_body(x)
        policy_logits = self.action_layer(actor).view(-1, self.option_dim, self.action_dim)
        betas = self.beta_layer(actor)
        values = self.value_layer_options(x)
        return policy_logits, betas, values

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, option_dim, hidden_size=128):
        super().__init__()
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
    def __init__(self, model, decay=0.1, step_alpha=0.1):
        self.model = model
        self.action_dim = model.action_dim
        self.option_dim = model.option_dim
        self.batch = None
        self.dim = None
        self.omega = deque(maxlen=2000)
        # switch matrix which control number of options
        # mtcts part:
        self.decay = decay
        self.step_alpha = step_alpha
        self.average_reward = 0
        self.preference = torch.zeros(self.option_dim).to(device)
        self.prob = F.softmax(self.preference, dim=-1)
        self.fixed_options = torch.arange(self.option_dim).to(device).long()

        # self.epsilon = 1
        # self.epsilon_count = 1/epsilon_last
        # self.episodes_to_play = np.zeros(self.option_dim) + 10
        # self.fixed_options = torch.arange(self.option_dim).long()
        # self.values = np.zeros(self.option_dim) + float('inf')
        # self.list = []

    def update_mab(self, value):
        '''Update MAB after reaching termination state.'''
        curr_node = len(self.fixed_options) - 1
        self.average_reward = (1-self.decay)*self.average_reward + self.decay*value
        self.preference -= self.step_alpha*(value - self.average_reward)*self.prob
        self.preference[curr_node] += self.step_alpha*(value - self.average_reward)
        self.prob = F.softmax(self.preference/torch.abs(self.preference).max().item(), dim=-1)
        option = Categorical(self.prob).sample().item()
        self.fixed_options = torch.arange(option+1).to(device).long()
        # curr_node = len(self.fixed_options) - 1
        # self.list.append(value)
        # if len(self.list) == 10:
        #     self.values[curr_node] = np.mean(self.list)
        #     greedy_option = np.argmax(self.values)
        #     random_option = np.random.randint(0, self.option_dim)
        #     option = random_option if np.random.rand() < self.epsilon else greedy_option
        #     self.fixed_options = torch.arange(option+1).long()
        #     self.list = []
        # self.epsilon = max(0.01, self.epsilon-self.epsilon_count)

    def options_eps_greedy(self, q_omg, epsilon=0.1):
        # create mask to choose between eps_greedy and argmax strategies:
        size = (1,) if self.dim == 1 else (self.batch, 1)
        mask = (torch.rand(*size) < epsilon).to(device).long()
        # make customizable options:
        if self.fixed_options is not None:
            idx_rand = torch.randint(len(self.fixed_options), size).to(device)
            idx_max = torch.argmax(q_omg[:,self.fixed_options], dim=-1, keepdim=True)
            idx_max = self.fixed_options[idx_max]
        else:
            idx_rand = torch.randint(self.option_dim, size).to(device)
            idx_max = torch.argmax(q_omg, dim=-1, keepdim=True)
        return mask*idx_rand + (1-mask)*idx_max

    def act(self, inputs, options=None, restart=None, epsilon=0.01, training=False):
        """return values corresponding to given options and return new options"""
        x = torch.tensor(inputs, dtype=torch.float32).to(device)
        self.batch = 1 if x.dim()==1 else x.shape[0]
        self.dim = x.dim()
        policy_logits, betas, q_omg = self.model(x)
        # create mask to sample new options if terminates:
        mask = (betas.gather(-1, options) > torch.rand(*options.size()).to(device)).long() # terminates
        # start new options for new episode:
        # find idx to restart:
        idx_restart = (restart == 1).nonzero()
        # make sure that restart episode have new options
        mask[idx_restart] = 1
        new_options = self.options_eps_greedy(q_omg, epsilon)
        options = (1-mask)*options + mask*new_options
        # apply options to policies:
        self.omega.append(options.max().item())
        policy_logits_omg = policy_logits[range(self.batch), options.squeeze()]
        dist = Categorical(logits=policy_logits_omg)
        return {"distribution": dist,
                "options": options,
                "betas": betas,
                "q_omg": q_omg}


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
        self.options = torch.zeros(len(self.envs), 1).to(device).long()
        self.prev_options = torch.zeros(len(self.envs), 1).to(device).long()
        self.restart = torch.ones(len(self.envs), 1).to(device).long()
        # Whether particular session has just been terminated and needs restarting.
        self.just_ended = [0] * len(self.envs)
        self.total_reward = [0] * len(self.envs)

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
                self.total_reward[i] += cur_reward
                if is_done:
                    # Game ends now, will finalize on next tick.
                    self.just_ended[i] = 1
                    self.restart[i] = 1
                    self.policy.update_mab(self.total_reward[i])
                    self.total_reward[i] = 0
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
            self.restart *= 0
            actions = act['distribution'].sample()
            new_obs, cur_rwds, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions.detach().cpu().numpy()))
            # Append data tuple for this tick
            cur_rwds_t = torch.tensor(cur_rwds, dtype=torch.float32).to(device)[:,None]
            dist = act['distribution']
            betas = act['betas']
            q_omg = act['q_omg']
            history_log.append((self.prev_observations,
                                actions,
                                cur_rwds_t,
                                self.restart,
                                self.options,
                                self.prev_options,
                                dist,
                                betas,
                                q_omg))
            self.prev_observations = new_obs
            self.prev_options = self.options
        # cast to numpy arrays, transpose from [time, batch, ...] to [batch, time, ...]
        history_log = list(reversed(history_log)) # reverse time
        # history_log = [np.array(tensor).swapaxes(0, 1) for tensor in zip(*history_log)]
        # history_log = [tuple(map(np.asarray, one_pass)) for one_pass in history_log]
        # observation_seq, action_seq, reward_seq, is_alive_seq = history_log
        trajectory = {'history_log': history_log}
        trajectory['last_state'] = self.prev_observations
        trajectory['last_option'] = self.prev_options
        trajectory['last_restart'] = self.restart
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
        self.gradient_clip = 0.5

    def loss(self, trajectory):
        L_pi = []
        E = []
        L_beta = []
        L_critic = []
        predictions = self.policy.act(trajectory['last_state'],
                                      trajectory['last_option'],
                                      trajectory['last_restart'])
        last_options = trajectory['last_option']
        beta_last = predictions['betas'].gather(1, last_options)
        q_omg_last = predictions['q_omg']
        return_util = (1-beta_last)*q_omg_last.gather(1, last_options) \
                      + beta_last*torch.max(q_omg_last, dim=1, keepdim=True)[0]
        return_util = return_util.detach()
        iterate = trajectory['history_log']
        for state, action, reward, done, option, prev_option, dist, beta, q_omg in iterate:
            # calculate return for util function:
            return_util = reward + self.gamma*return_util*(1-done.float())
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
            L_beta.append(beta.gather(1, prev_option)*adv_beta.detach()*(1-done.float()))
        policy_loss = torch.stack(L_pi, dim=1).to(device).mean()
        entropy_loss = torch.stack(E, dim=1).to(device).mean()
        beta_loss = torch.stack(L_beta, dim=1).to(device).mean()
        critic_loss = torch.stack(L_critic, dim=1).to(device).mean()
        loss = -policy_loss + beta_loss + self.critic_loss_coef*critic_loss# - self.entropy_coef*entropy_loss
        return loss, (policy_loss, critic_loss, entropy_loss)

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step."""
        loss, info_data = self.loss(trajectory)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, info_data


def evaluate_options(policy, env, n_games=1, show=False):
    """Plays an entire game start to end, returns session rewards."""
    game_rewards = []
    for _ in range(n_games):
        # initial observation
        options = torch.zeros(1, 1).to(device).long()
        restart = torch.ones(1, 1).to(device).long()
        observation = env.reset()
        total_reward = 0
        while True:
            if show:
                env.render()
            act = policy.act([observation], options, restart)
            restart = torch.zeros(1, 1).to(device).long()
            actions = act['distribution'].sample().detach().cpu().numpy()[0]
            options = act['options']
            if show:
                print(f"Options: {options.item()}; actions: {actions.item()}")
            observation, reward, done, info = env.step(actions)
            total_reward += reward
            if done:
                if show:
                    env.close()
                break
        game_rewards.append(total_reward)
    return game_rewards

def train_oc(policy, optimizer, make_env, num_iter=100, upper_limit=float('inf')):
    epsilon = (max(0.99995**i, 0.01) for i in range(num_iter))
    oc = OC(policy, optimizer, beta_reg = 0.01)
    pool = EnvOptionPool(make_env, policy, n_parallel_games=1)
    for i in range(num_iter):
        eps = next(epsilon)
        oc.step(pool.get_next(10, epsilon=eps))
        if i%100 == 0:
            # print(Counter(policy.omega), f"\nEpsl: {eps:.4f}")
            # print("Prob: ", policy.values.round())
            reward = np.mean(evaluate_options(policy, env, n_games=4))
            # evaluate_options(policy, env, show=True)
            print(f'Prob: {policy.prob.detach().cpu().numpy().round(3)}')
            print(f'Pref: {policy.preference.detach().cpu().numpy().round(1)}')
            print(f'Rwrd: {reward:.1f}, Epis: {Summaries.number_of_episodes}', end=' ')
            print(f'AvgR: {policy.average_reward:.1f},BstO: {len(policy.fixed_options)}')
            if reward > upper_limit:
                print("Env is solved! Reward: ", reward)
                break

if __name__ == '__main__':
    from utils import *
    from functools import partial

    env = make_env('CarIntersect-v1', 'switch_oc')
    func_env = partial(make_env, 'CarIntersect-v1', 'switch_oc')
    # env = make_env('LunarLander-v2', 'switch_oc_2')
    # func_env = partial(make_env, 'LunarLander-v2', 'switch_oc_2')
    # env = make_env('CartPole-v0', 'switch_oc')
    # func_env = partial(make_env, 'CartPole-v0', 'switch_oc')
    # env = make_env('Asterix-ram-v0', 'switch')
    # func_env = partial(make_env, 'Asterix-ram-v0', 'switch')
    # env = make_pixel_env('Asterix-v0', 'switch_oc')
    # func_env = partial(make_pixel_env, 'Asterix-v0', 'switch_oc')

    model = Model(env.observation_space.shape[0], env.action_space.n, option_dim=10, hidden_size=256).to(device)
    # model = ConvModel(env.observation_space.shape, env.action_space.n, option_dim=10, hidden_size=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy = Policy(model, decay=0.1, step_alpha=0.4)

    train_oc(policy, optimizer, func_env, num_iter=100000, upper_limit=50)
    print('Optimal options: ', policy.fixed_options)
    print("Prob: ", policy.preference.round())

    evaluate_options(policy, env, show=True)
