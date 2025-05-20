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
import matplotlib
matplotlib.use('Agg')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__=="__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for dual lagrangian network")
    # --------------------------------------Hyperparameter--------------------------------------------------------------------
    parser.add_argument("--policy_name", type=str, default='TD3', help="TD3、DDPG")
    parser.add_argument("--buffer_size", type=int, default=int(1e4), help="Number of training samples")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Maximum number of episodes")
    parser.add_argument("--num_step", type=int, default=15, help="Maximum number of episodes")
    parser.add_argument("--tau", type=float, default=1e-5, help=" ")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of actor newtork")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimesion of hidden layers")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount factor")
    parser.add_argument("--lmbda", type=float, default=0.8, help=" ")

    parser.add_argument("--minimal_size", type=int, default=500, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size,128")
    parser.add_argument("--decay", default=1e-5, type=float, metavar='G',help='Decay rate for the networks (default: 0.00001)')

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
    # parser.add_argument("--sigma_t", default=1e-10, type=float, metavar='G',help='reflection coefficient of the target (default: 0.01)') #check
    parser.add_argument("--Pbs", default=1.0, type=float, metavar='N', help='transmit power at ISAC-BS (W)')
    parser.add_argument("--H", default=2, type=int, metavar='N',help='height of the BS, target and IRS (default: 10 m)')
    parser.add_argument("--Gam", default=1e-3, type=float, metavar='N', help='communication rate threshold (default: 0.1 bps)') #check
    parser.add_argument("--large_loss", default=1e-3, type=float, metavar='N', help='large-scale fading')
    args = parser.parse_args()
    args.Ns = int(args.Ny * args.Nz)
    # Set random seed
    set_seed(6)
    args.lamda = 3e8 / args.fc  # waveform length
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    Js = JakeModelV1(args.lamda, args.Wz, args.Wy, args.Nz, args.Ny, args.large_loss)    #spatial coefficient
    b = list(range(args.Ns))
    combinations = get_combinations_recursive(b, args.ns)
    args.combinations = combinations
    print(device)

    # create communication-only environment
    env = environment.ISAC(args, Js)
    env_name = 'isac'
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1
    file_name = f"{args.num_step}_{args.lr}_{args.decay}"
    replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, device, max_size=args.buffer_size)
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
    w_opt = np.zeros((args.Ns, 1), dtype=complex)
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
            next_state, reward, w, done, _ = env.step(action)

            # show_beampattern(args.num_antennas, args.num_snapshots, theta, action, reward)

            done = 1.0 if t == args.num_step - 1 else float(done)

            # next_state = whiten(next_state)

            # episode_reward += reward

            # save optimal solution
            if reward > max_reward:
                max_reward = reward
                action_opt = action[0,-1]
                w_opt = w

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

    # if args.save_model:
    #     agent.save(f"./Models/{file_name}")

    end_time = time.time()
    print('time = ' + str(end_time - start_time))  # time (hour)

    # # plot convergence curve
    # plt.figure()
    # plt.plot(instant_rewards, 'r-')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.grid(True)
    # name = args.policy_name + '_convergence.png'
    # plt.savefig(name)
    # plt.close()
    # print(str(np.max(instant_rewards)))
    # print(name)

    # plt.figure()
    # plt.plot(instant_rewards, 'r-')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.grid(True)
    # plt.show()

    bins = np.linspace(-1, 1, len(combinations))
    max_index = combinations[np.digitize(action_opt, bins) - 1]
    # max_index = combinations[indexes]
    # show_nodes(max_index, args.Ny, args.policy_name)
    np.savez(f"./Learning Curves/{args.policy_name}/{file_name}_episode_{episode_num + 1}_{args.policy_name}.npz",return_list = instant_rewards,
             max_index = max_index, max_reward = max_reward, w_opt = w_opt)

    # np.savez('./results/' + name + '.npy', return_list = return_list, max_index = max_index)
    # print(str(np.max(instant_rewards)))
    print(args.policy_name,max_reward)