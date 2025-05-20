from datetime import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import argparse
import environment
from utils import *
import os
import torch.nn.init as init
import torch.nn as nn
import time
import random
import matplotlib
matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# orthogonal init
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
# actor network
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.ln = torch.nn.LayerNorm(state_dim)
        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x):
        x = self.ln(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).clamp(min=-50, max=50)
        return F.softmax(x, dim=1)

#critic network
class ValueNetGAN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetGAN, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 1, hidden_dim)

        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.ln = torch.nn.LayerNorm(state_dim)
        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

    def forward(self, x, noise):
        x = self.ln(x)
        x1 = torch.cat([x,  noise], 1)
        x1 = F.relu(self.fc1(x1))
        return self.fc2(x1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.ln = torch.nn.LayerNorm(state_dim)
        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

    def forward(self, x):
        x = self.ln(x)
        # x1 = torch.cat([x,  noise], 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Discriminator(torch.nn.Module):

    def __init__(self, ):
        super(Discriminator, self).__init__()

        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, 0.01)
    def forward(self, action_value):
        # state_action = torch.cat([action_value, action_value_target], 1)
        # state_action = action_value

        q = F.relu(self.fc1(action_value))
        q = F.relu(self.fc2(q))
        q = torch.tanh(self.fc3(q))
        return q

class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim,  action_dim, args, device):

        self.actor = PolicyNet(state_dim, args.hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, args.hidden_dim).to(device)
        # self.discriminator = Discriminator().to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.decay)
        # self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.critic_lr, weight_decay=args.decay)
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.epochs = args.epochs  # 一条序列的数据用来训练轮数
        self.eps = args.eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # print(probs)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)

        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # noises = torch.ones_like(rewards).data.normal_(0, 1).to(device) / 10# todo
        current_Q = self.critic(states)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_delta = td_target - current_Q

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()
        # loss_discriminator_Q = -torch.mean(torch.log(self.discriminator(current_Q.detach())) + torch.log(1 - self.discriminator(td_target)))
        #
        # self.discriminator_optimizer.zero_grad()
        # loss_discriminator_Q.backward()
        # self.discriminator_optimizer.step()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # dist_entropy =  torch.distributions.Categorical(probs=self.actor(states)).entropy().view(-1, 1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage # 截断

            actor_loss = torch.mean(-torch.min(surr1, surr2))   # PPO损失函数

            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach())) #todo

            self.actor_optimizer.zero_grad()

            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()

            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

class GANPPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim,  action_dim, args, device):

        self.actor = PolicyNet(state_dim, args.hidden_dim, action_dim).to(device)
        self.critic = ValueNetGAN(state_dim, args.hidden_dim).to(device)
        self.discriminator = Discriminator().to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.decay)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.critic_lr) #3*args.critic_lr
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.epochs = args.epochs  # 一条序列的数据用来训练轮数
        self.eps = args.eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # print(probs)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)

        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        noises = torch.ones_like(rewards).data.normal_(0, 0.5).to(device)  # /5,todo
        current_Q = self.critic(states, noises)
        td_target = rewards + self.gamma * self.critic(next_states, noises) * (1 - dones)

        td_delta = td_target - current_Q

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()
        loss_discriminator_Q = -torch.mean(torch.log(self.discriminator(current_Q.detach())) + torch.log(1 - self.discriminator(td_target)))

        self.discriminator_optimizer.zero_grad()
        loss_discriminator_Q.backward()
        self.discriminator_optimizer.step()

        for _ in range(self.epochs):
            # noises = torch.ones_like(rewards).data.normal_(0, 1).to(device) / 15
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # dist_entropy =  torch.distributions.Categorical(probs=self.actor(states)).entropy().view(-1, 1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage # 截断

            actor_loss = torch.mean(-torch.min(surr1, surr2)) # PPO损失函数

            critic_loss = torch.mean(F.mse_loss(self.critic(states, noises), td_target.detach())) -torch.mean(torch.log(1 - self.discriminator(td_target.detach())))#todo

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



if __name__=="__main__":
    ns_all = [3,4, 5, 6]
    mento_carlo = 2e1
    max_value_all = []
    for ns_tmp in ns_all:

        max_value_sum = 0
        for ii in range(int(mento_carlo)):
            parser = argparse.ArgumentParser("Hyperparameters Setting for dual lagrangian network")
            # --------------------------------------Hyperparameter--------------------------------------------------------------------
            # parser.add_argument("--n_samples", type=int, default=int(1e4), help="Number of training samples")
            parser.add_argument("--agent_name", type=str, default='GANPPO', help="GANPPO, PPO")
            parser.add_argument("--num_episodes", type=int, default=1000, help="Maximum number of episodes")
            parser.add_argument("--epochs", type=int, default=20, help=" ")
            parser.add_argument("--actor_lr", type=float, default=1e-5, help="learning rate of actor newtork,1e-5")
            parser.add_argument("--critic_lr", type=float, default=1e-4, help="learning rate of actor newtork")
            parser.add_argument("--hidden_dim", type=int, default=128, help="dimesion of hidden layers")
            parser.add_argument("--gamma", type=float, default=0.9, help="discount factor， 0.95")
            parser.add_argument("--lmbda", type=float, default=0.9, help=" ")
            parser.add_argument("--eps", type=float, default=0.4, help="PPO中截断范围的参数")
            # parser.add_argument("--buffer_size", type=int, default=int(1e5), help="The capacity of the replay buffer")
            # parser.add_argument("--batch_size", type=int, default=32, help="Batch size,128")
            parser.add_argument("--width_p", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
            parser.add_argument("--width_d", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
            parser.add_argument("--depth_p", type=int, default=4, help="depth of the neural network")
            parser.add_argument("--depth_d", type=int, default=4, help="depth of the neural network")
            parser.add_argument("--decay", default=1e-5, type=float, metavar='G',help='Decay rate for the networks (default: 0.00001)')


            # parser.add_argument("--noise_std_init", type=float, default=0.5, help="The std of Gaussian noise for exploration, 0.5")
            # parser.add_argument("--noise_std_min", type=float, default=0.2, help="The std of Gaussian noise for exploration")
            # parser.add_argument("--noise_decay_steps", type=float, default=4e3, help="How many steps before the noise_std decays to the minimum")
            # parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
            # parser.add_argument("--lr_p", type=float, default=1e-5, help="Learning rate of primary network")
            # parser.add_argument("--lr_d", type=float, default=1e-5, help="Learning rate of dual network")
            # parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
            # parser.add_argument("--tau", type=float, default=1e-5, help="Softly update the target network")
            # parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
            # parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
            # parser.add_argument("--penalty", type=float, default=10000.0, help="penalty factor for the constraints")
            # parser.add_argument("--penalty_F", type=float, default=1000.0, help="penalty factor for the objective function")
            # --------------------------------------System-specific parameters--------------------------------------------------------------------
            parser.add_argument("--fc", default=3.4e9, type=float, metavar='N', help='frequency')
            parser.add_argument("--Wy", default=0.1, type=float, metavar='N', help='length along y-axis')
            parser.add_argument("--Wz", default=0.1, type=float, metavar='N', help='length along z-axis')
            parser.add_argument("--Ny", default=6, type=int, metavar='N', help='number of ports along Wy')
            parser.add_argument("--Nz", default=6, type=int, metavar='N', help='number of ports along Wz')
            parser.add_argument("--ns", default=5, type=int, metavar='N', help='number of activated ports')
            parser.add_argument("--zeta", default=1.0, type=float, metavar='N', help='Target RCS')
            # parser.add_argument("--Nk", default=2, type=int, metavar='N', help='Number of antennas in the users')
            # parser.add_argument("--L", default=30, type=int, metavar='N', help='Number of Snapshots')
            # parser.add_argument("--J", default=4, type=int, metavar='N', help='Number of ISAC-BS')
            parser.add_argument("--K", default=2, type=int, metavar='N', help='Number of communication users')
            parser.add_argument("--sigma_n", default=1e-10, type=float, metavar='G',help='Variance of the additive white Gaussian noise (default: -110dBm)')
            parser.add_argument("--sigma_t", default=1e-10, type=float, metavar='G',help='reflection coefficient of the target (default: 0.01)') #check
            parser.add_argument("--Pbs", default=0.1, type=float, metavar='N', help='transmit power at ISAC-BS (W)')
            parser.add_argument("--H", default=2, type=int, metavar='N',help='height of the BS, target and IRS (default: 10 m)')
            parser.add_argument("--Gam", default=0.1, type=float, metavar='N', help='communication rate threshold (default: 0.1 bps)') #check
            parser.add_argument("--large_loss", default=1e-3, type=float, metavar='N', help='large-scale fading')
            parser.add_argument("--iter_num", default=15, type=int, metavar='N', help='number of iterations, 20')
            args = parser.parse_args()
            args.Ns = int(args.Ny * args.Nz)
            args.ns = ns_tmp
            # Set random seed
            set_seed(ii)
            wave_length = 3e8 / args.fc  # waveform length
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")
            Js = JakeModelV1(wave_length, args.Wz, args.Wy, args.Nz, args.Ny, args.large_loss)    #spatial coefficient
            b = list(range(args.Ns))
            combinations = get_combinations_recursive(b, args.ns)
            args.combinations = combinations
            print(device)
            # create communication-only environment
            env = environment.COM(args, Js)
            env_name = 'com_only'
            state_dim = env.state_dim
            action_dim = env.action_dim

            if args.agent_name == 'GANPPO':
                agent = GANPPO(state_dim, action_dim, args, device)
            else:
                agent = PPO(state_dim, action_dim, args, device)

            begin_time = time.time()
            return_list, max_action, max_reward, _ = rl_utils.train_on_policy_agent(env, agent, args.num_episodes, args.iter_num)

            max_value_sum = max_value_sum + max_reward

        max_value_all.append(max_value_sum / mento_carlo)

    np.save(args.agent_name + '_reward_vs_ns.npy', max_value_all)