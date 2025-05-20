import random
import time
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import rl_utils
import argparse
import environment
import TD3
import DDPG
from utils import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# torch.cuda.set_device(1)  # 设置默认 GPU 为 1

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
    mento_carlo = 1e2
    max_value_all = []
    for ii in range(int(mento_carlo)):
        parser = argparse.ArgumentParser("Hyperparameters Setting for dual lagrangian network")
        # --------------------------------------Hyperparameter--------------------------------------------------------------------
        parser.add_argument("--policy_name", type=str, default='TD3', help="TD3、DDPG")
        parser.add_argument("--buffer_size", type=int, default=int(1e4), help="Number of training samples")
        parser.add_argument("--num_episodes", type=int, default=1000, help="Maximum number of episodes")
        parser.add_argument("--num_step", type=int, default=15, help="Maximum number of episodes")
        parser.add_argument("--tau", type=float, default=1e-5, help=" ")
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of actor newtork")
        parser.add_argument("--hidden_dim", type=int, default=128, help="dimesion of hidden layers")
        parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
        parser.add_argument("--lmbda", type=float, default=0.9, help=" ")

        parser.add_argument("--minimal_size", type=int, default=500, help="The capacity of the replay buffer")
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size,128")
        # parser.add_argument("--width_p", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
        # parser.add_argument("--width_d", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
        # parser.add_argument("--depth_p", type=int, default=4, help="depth of the neural network")
        # parser.add_argument("--depth_d", type=int, default=4, help="depth of the neural network")
        parser.add_argument("--decay", default=1e-5, type=float, metavar='G',help='Decay rate for the networks (default: 0.00001)')

        # parser.add_argument("--target_update", type=int, default=10, help="The std of Gaussian noise for exploration, 0.5")
        # parser.add_argument("--epsilon", type=float, default=0.1, help="The std of Gaussian noise for exploration")
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

        parser.add_argument("--K", default=2, type=int, metavar='N', help='Number of communication users')
        parser.add_argument("--sigma_n", default=1e-10, type=float, metavar='G',help='Variance of the additive white Gaussian noise (default: -110dBm)')
        parser.add_argument("--sigma_t", default=1e-10, type=float, metavar='G',help='reflection coefficient of the target (default: 0.01)') #check
        parser.add_argument("--Pbs", default=0.1, type=float, metavar='N', help='transmit power at ISAC-BS (W)')
        parser.add_argument("--H", default=2, type=int, metavar='N',help='he ight of the BS, target and IRS (default: 10 m)')
        parser.add_argument("--Gam", default=0.1, type=float, metavar='N', help='communication rate threshold (default: 0.1 bps)') #check
        parser.add_argument("--large_loss", default=1e-3, type=float, metavar='N', help='large-scale fading')
        # parser.add_argument("--iter_num", default=15, type=int, metavar='N', help='number of iterations')
        args = parser.parse_args()
        args.Ns = int(args.Ny * args.Nz)
        # Set random seed
        set_seed(ii)
        wave_length = 3e8 / args.fc  # waveform length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda:1")
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
        max_action = 1
        file_name = f"{args.num_step}_{args.lr}_{args.decay}"
        replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
        print(args.policy_name)
        # Initialize the algorithm
        if args.policy_name == 'TD3':
            agent = TD3.TD3(state_dim, action_dim, args, device)
        else:
            agent = DDPG.DDPG(state_dim, action_dim, args, device)
        # Initialize the instant rewards recording array
        instant_rewards = []

        max_reward = 0
        action_opt = 0
        explore_niose = 0.6
        start_time = time.time()
        for eps in range(int(args.num_episodes)):

            state, done = env.reset(), False
            # episode_reward = 0
            episode_num = 0
            episode_time_steps = 0

            # state = whiten(state)
            if explore_niose >= 0.2:
                explore_niose = explore_niose * 0.999

            eps_rewards = 0

            for t in range(int(args.num_step)):
                # Choose action from the policy
                action = agent.select_action(np.array(state)).reshape(1, -1)

                # # # # #add noise
                noise = np.random.normal(0, explore_niose * max_action, size=action_dim).reshape(1, -1).clip(-0.2 * max_action, 0.2 * max_action)

                action = (action + noise)

                # Take the selected action
                next_state, reward, done, _ = env.step(action)

                # show_beampattern(args.num_antennas, args.num_snapshots, theta, action, reward)

                done = 1.0 if t == args.num_step - 1 else float(done)

                # next_state = whiten(next_state)

                # episode_reward += reward

                # save optimal solution
                if reward > max_reward:
                    max_reward = reward
                    action_opt = action[0,-1]

                # reward = reward - np.mean(eps_rewards)

                # change reward
                # reward = reward

                eps_rewards += reward

                # Store data in the experience replay buffer
                replay_buffer.add(state, action, next_state, reward, done)

                state = next_state
                if replay_buffer.size >= args.batch_size:
                    # Train the agent
                    agent.update_parameters(replay_buffer, args.batch_size)

                # print(f"Time step: {t + 1} Episode Num: {eps + 1} Reward: {reward:.3f}")

                episode_time_steps += 1

                if done:
                    print(f"Total T: {t + 1} Episode Num: {eps + 1} Episode T: {episode_time_steps} Eps Reward: {eps_rewards}")

                    # Reset the environment
                    state, done = env.reset(), False
                    episode_reward = 0
                    episode_time_steps = 0
                    episode_num += 1

                    # state = whiten(state)

                    instant_rewards.append(eps_rewards)

                    # np.save(f"./Learning Curves/{args.policy_name}/{file_name}_episode_{episode_num + 1}_{args.policy_name}", instant_rewards)

 

        end_time = time.time()
        print('time = ' + str(end_time - start_time))  # time (hour)
        
        max_value_all.append(max_reward/ mento_carlo)
        
    np.save(args.policy_name + '_mentocarlo' + '.npy', max_value_all)
        
