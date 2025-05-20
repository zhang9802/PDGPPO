import random
# import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import argparse
import environment
from utils import *
import os
import matplotlib
matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action (action_dim - 1)) * (action_upbound -action_lowbound)



os.environ['KMP_DUPLICATE_LIB_OK']='True'
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 隐藏层使用ReLU激活函数
        return self.fc3(x)

class DQN:
    ''' DQN算法,包括Double DQN '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 decay,
                 device,
                 dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate,weight_decay=decay)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)

        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)


        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络

        self.count += 1

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    max_action = 0
    max_reward = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),desc='Iteration %d' % i) as pbar:

            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                for _ in range(args.iter_num):
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理

                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    # action_continuous = dis_to_con(action, env,
                    #                                agent.action_dim)
                    next_state, reward, done, _ = env.step(action)
                    if reward > max_reward:
                        max_action = action
                        max_reward = reward
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)

                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list, max_action, max_reward


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
    parser = argparse.ArgumentParser("Hyperparameters Setting for dual lagrangian network")
    # --------------------------------------Hyperparameter--------------------------------------------------------------------
    parser.add_argument("--buffer_size", type=int, default=int(1e4), help="Number of training samples")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Maximum number of episodes")
    parser.add_argument("--epochs", type=int, default=20, help=" ")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of actor newtork")
    # parser.add_argument("--critic_lr", type=float, default=1e-4, help="learning rate of actor newtork")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimesion of hidden layers")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount factor")
    parser.add_argument("--lmbda", type=float, default=0.8, help=" ")

    parser.add_argument("--minimal_size", type=int, default=200, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size,128")
    parser.add_argument("--width_p", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--width_d", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--depth_p", type=int, default=4, help="depth of the neural network")
    parser.add_argument("--depth_d", type=int, default=4, help="depth of the neural network")
    parser.add_argument("--decay", default=1e-5, type=float, metavar='G',help='Decay rate for the networks (default: 0.00001)')

    parser.add_argument("--target_update", type=int, default=10, help="The std of Gaussian noise for exploration, 0.5")
    parser.add_argument("--epsilon", type=float, default=0.05, help="The std of Gaussian noise for exploration")

    # --------------------------------------System-specific parameters--------------------------------------------------------------------
    parser.add_argument("--fc", default=3.4e9, type=float, metavar='N', help='frequency')
    parser.add_argument("--Wy", default=0.1, type=float, metavar='N', help='length along y-axis')
    parser.add_argument("--Wz", default=0.1, type=float, metavar='N', help='length along z-axis')
    parser.add_argument("--Ny", default=6, type=int, metavar='N', help='number of ports along Wy')
    parser.add_argument("--Nz", default=6, type=int, metavar='N', help='number of ports along Wz')
    parser.add_argument("--ns", default=5, type=int, metavar='N', help='number of activated ports')
    parser.add_argument("--zeta", default=1.0, type=float, metavar='N', help='Target RCS')
    parser.add_argument("--K", default=2, type=int, metavar='N', help='Number of users')
    parser.add_argument("--J", default=2, type=int, metavar='N', help='Number of targets')
    parser.add_argument("--sigma_n", default=1e-10, type=float, metavar='G',help='Variance of the additive white Gaussian noise (default: -110dBm)')

    parser.add_argument("--Pbs", default=1.0, type=float, metavar='N', help='transmit power at ISAC-BS (W)')
    parser.add_argument("--H", default=2, type=int, metavar='N',help='height of the BS, target and IRS (default: 10 m)')
    parser.add_argument("--Gam", default=1e-3, type=float, metavar='N', help='communication rate threshold (default: 0.1 bps)') #check
    parser.add_argument("--large_loss", default=1e-3, type=float, metavar='N', help='large-scale fading')
    parser.add_argument("--iter_num", default=15, type=int, metavar='N', help='number of iterations')
    args = parser.parse_args()
    args.Ns = int(args.Ny * args.Nz)
    # Set random seed
    set_seed(0)
    args.lamda =  3e8 / args.fc  # waveform length
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    Js = JakeModelV1(args.lamda, args.Wz, args.Wy, args.Nz, args.Ny, args.large_loss)    #spatial coefficient
    b = list(range(args.Ns))
    combinations = get_combinations_recursive(b, args.ns)
    args.combinations = combinations
    print(device)
    replay_buffer = ReplayBuffer(args.buffer_size)
    # create communication-only environment
    env = environment.ISAC(args, Js)
    env_name = 'isac'
    state_dim = env.state_dim
    action_dim = env.action_dim
    dqn_type = 'VanillaDQN' # 'VanillaDQN', 'DoubleDQN'
    agent = DQN(state_dim, action_dim, args.hidden_dim, args.lr, args.gamma, args.epsilon,args.target_update, args.decay, device,dqn_type)

 
    return_list, max_q_value_list, max_action, max_reward = train_DQN(agent, env, args.num_episodes, replay_buffer, args.minimal_size, args.batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 9)
    name = dqn_type + '_lr_' + str(args.lr) +  '_hid_' + str(args.hidden_dim)  + '_gamma_' + str(args.gamma) + '_bs_' + str(args.batch_size)

    print(max_reward)

    plt.figure()
    plt.plot(episodes_list, return_list,'b', alpha=0.3,label='reward')
    plt.plot(episodes_list, mv_return,'b',label='smoothed reward')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    plt.legend()
    plt.title(dqn_type + ', max_reward=' + str(max_reward))
    plt.savefig(name + '.png', dpi=900)

    # print(name)
    # plt.ylabel('Returns')
    # plt.grid()
    # plt.legend()
    # plt.show()

    # print(name)
    max_index = combinations[max_action]
    # show_nodes(max_index, args.Ny, 'pos_' + name + '.png')
    np.savez('./results/' + name + '.npz', return_list = return_list, max_index = max_index, max_reward = max_reward)