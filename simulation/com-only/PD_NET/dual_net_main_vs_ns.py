#data: 250315
#author: jifa zhang
#version: v1
#description: dual-lagrangian-based method for constrained optimization
import argparse
import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
from utils import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# create the class of primal network
class ResNet(torch.nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        # self.params = params
        self.linearIn = nn.Linear(args.Ns * args.K * 2, args.width_p) # input layer
        self.batchNorm = nn.BatchNorm1d(args.Ns * args.K * 2)
        self.linear = nn.ModuleList()
        for _ in range(args.depth_p):
            self.linear.append(nn.Linear(args.width_p, args.width_p)) # hidden layers

        self.linearOut = nn.Linear(args.width_p, args.Ns *2) # output layer

    def forward(self, x):

        # x = self.batchNorm(x)
        x = self.linearIn(x)  # input layer
        for layer in self.linear:
            x_temp = F.relu(layer(x)) # activation $\tanh^3$
            x = x_temp+x # residual connection

        x = self.linearOut(x) # output layer
        power = torch.linalg.norm(x, axis=1, keepdims=True).repeat(1, x.shape[1])
        # power1 = torch.sqrt(x[:,:args.Ns].detach()**2 + x[:,args.Ns:].detach()**2)
        # power = torch.cat((power1, power1), dim=1)
        x = x / power * np.sqrt(args.Pbs)
        return x

# create the class of adversarial network
class ResNet_1(torch.nn.Module):
    def __init__(self, args):
        super(ResNet_1, self).__init__()
        # self.params = params
        self.linearIn = nn.Linear(args.Ns * args.K * 2, args.width_d)
        self.batchNorm = nn.BatchNorm1d(args.Ns * args.K * 2)
        self.linear = nn.ModuleList()
        for _ in range(args.depth_d):
            self.linear.append(nn.Linear(args.width_d, args.width_d))

        self.linearOut = nn.Linear(args.width_d, args.K)

    def forward(self, x):
        # x = self.batchNorm(x)
        x = self.linearIn(x)
        for layer in self.linear:
            x_temp = F.relu(layer(x))
            x = x_temp + x

        x = self.linearOut(x)

        x = F.relu(x) # non-positivity constaint

        return x

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def cal_obj(H, x, args, device):
    """
    calculate objective function (communication achievable rate)
    :param H:
    :param x:
    :param args:
    :param device:
    :return:
    """
    H_real, H_imag = H[:,:args.Ns * args.K].reshape(args.batch_size, args.K, args.Ns), H[:,args.Ns * args.K:].reshape(args.batch_size, args.K, args.Ns)
    Rk = torch.zeros((args.batch_size, args.K), dtype=torch.float32).to(device)
    for i in range(args.batch_size):
        for k in range(args.K):
            H_temp = torch.cat((H_real[i,k,:].reshape(-1,1).unsqueeze(0), H_imag[i,k,:].reshape(-1,1).unsqueeze(0)), dim=0)
            x_temp = torch.cat((x[i,:args.Ns].reshape(-1,1).unsqueeze(0), x[i, args.Ns:].reshape(-1,1).unsqueeze(0)), dim=0)
            Rk[i,k] = torch.log2(1 + torch.linalg.norm(cmul(conjT(H_temp), x_temp),'fro')**2 / args.sigma_n)

    return Rk

def train_net(model_p, device, args, optimizer_p,  model_d, optimizer_d, H):
    # train mode
    model_p.train()
    model_d.train()

    train_losses, valid_losses = [], []
    steps_per_epoch = args.n_samples // args.batch_size

    for i_epoch in range(args.n_epoch):
        indexes = np.random.permutation(args.n_samples)
        H_shuffeld = H[indexes]

        for step in range(steps_per_epoch):
            H_shuffeld_batch = H_shuffeld[step * args.batch_size:(step + 1) * args.batch_size]

            for iter in range(args.iter_num):
                x = model_p(H_shuffeld_batch)
                model_p.zero_grad()
                model_d.zero_grad()

                lambda_1 = model_d(H_shuffeld_batch)  # adversarial term
                # print("lambda_1", lambda_1)
                Rk = cal_obj(H_shuffeld_batch, x, args, device)
                loss1 = torch.mean(args.penalty_F / Rk)
                g = args.Gam - Rk
                loss2 = torch.mean(lambda_1 * g) + args.penalty/2 * torch.mean(torch.where(g > 0, g, 0)**2)
                loss_p = loss1 + loss2
                loss_d = - loss_p
                if iter%4 > 2:
                    loss_d.backward()
                    optimizer_d.step()
                else:
                    loss_p.backward()
                    optimizer_p.step()

            print(f"ns = {args.ns}, step = {step}, loss = {loss_p}")
            train_losses.append(loss_p.detach().item())

            # if step % 10 == 0:
            #     # plt.figure()
            #     plt.plot(train_losses)
            #     plt.xlabel('training steps')
            #     plt.ylabel('primary loss')
            #     plt.grid(True)
            #     name = 'convergence.png'
            #     plt.savefig(name, dpi=900)

    # save weights of primary network
    name = 'my_network_' + str(args.lr_p) + '_' + str(args.batch_size) + '_' + str(args.width_p) + '_' + str(args.ns) + '.pth'
    file_name = './weights/' + name
    torch.save(model_p.state_dict(), file_name)
    # plt.figure()
    plt.plot(train_losses)
    plt.xlabel('training steps')
    plt.ylabel('primary loss')
    plt.grid(True)
    name = 'convergence_lr_' + str(args.lr_p)
    plt.savefig(name + '.png', dpi=900)
    np.save(name + '.npy', train_losses)



if __name__=="__main__":
    ns_all = [3, 4]
    for ns_tmp in ns_all:
        parser = argparse.ArgumentParser("Hyperparameters Setting for dual lagrangian network")
        # --------------------------------------Hyperparameter--------------------------------------------------------------------
        parser.add_argument("--n_samples", type=int, default=int(1e4), help="Number of training samples")
        parser.add_argument("--n_epoch", type=int, default=20, help="Maximum number of episodes, L")
        # parser.add_argument("--evaluate_freq", type=float, default=100, help="Evaluate the policy every 'evaluate_freq' steps")
        # parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
        # parser.add_argument("--max_action", type=float, default=1.0, help="Max action")
        # parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
        # parser.add_argument("--buffer_size", type=int, default=int(1e5), help="The capacity of the replay buffer")
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size,128")
        parser.add_argument("--width_p", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--width_d", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--depth_p", type=int, default=4, help="depth of the neural network")
        parser.add_argument("--depth_d", type=int, default=4, help="depth of the neural network")
        parser.add_argument("--decay", default=1e-5, type=float, metavar='G',help='Decay rate for the networks (default: 0.00001)')


        # parser.add_argument("--noise_std_init", type=float, default=0.5, help="The std of Gaussian noise for exploration, 0.5")
        # parser.add_argument("--noise_std_min", type=float, default=0.2, help="The std of Gaussian noise for exploration")
        # parser.add_argument("--noise_decay_steps", type=float, default=4e3, help="How many steps before the noise_std decays to the minimum")
        # parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
        parser.add_argument("--lr_p", type=float, default=1e-4, help="Learning rate of primary network,1e-5")
        parser.add_argument("--lr_d", type=float, default=1e-4, help="Learning rate of dual network,1e-5")
        # parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
        # parser.add_argument("--tau", type=float, default=1e-5, help="Softly update the target network")
        parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
        parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
        parser.add_argument("--penalty", type=float, default=10000.0, help="penalty factor for the constraints")
        parser.add_argument("--penalty_F", type=float, default=1000.0, help="penalty factor for the objective function")
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
        parser.add_argument("--H", default=2, type=int, metavar='N',help='height of the BS, target and IRS (default: 10 m)')
        parser.add_argument("--Gam", default=0.1, type=float, metavar='N', help='communication rate threshold (default: 0.1 bps)') #check
        parser.add_argument("--large_loss", default=1e-3, type=float, metavar='N', help='large-scale fading')
        parser.add_argument("--iter_num", default=30, type=int, metavar='N', help='number of iterations')
        args = parser.parse_args()
        args.Ns = int(args.Ny * args.Nz)
        args.ns = ns_tmp
        # Set random seed
        torch.manual_seed(1)
        np.random.seed(1)
        lamda = 3e8 / args.fc  # waveform length
        Js = JakeModelV1(lamda, args.Wz, args.Wy, args.Nz, args.Ny, args.large_loss)
        device = torch.device("cpu")
        startTime = time.time()
        model_p = ResNet(args).to(device)
        model_d = ResNet_1(args).to(device)
        optimizer_p = torch.optim.Adam(model_p.parameters(), lr=args.lr_p, weight_decay=args.decay)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr_d, weight_decay=args.decay)

        # generate training samples, wireless channels
        train_data = generate_com_dataset(lamda, args.Wz, args.Wy, args.Nz, args.Ny, args.n_samples, args.K, args.Ns, args.ns, args.large_loss,Js)

        train_data = torch.from_numpy(train_data).float().to(device)  #numpy -> torch

        print("Generating network costs %s seconds."%(time.time() - startTime))
        startTime = time.time()
        train_net(model_p, device, args, optimizer_p,  model_d, optimizer_d, train_data)  #train primary_dual model
        print("Training costs %s seconds."%(time.time() - startTime))
        # name = 'my_network_' + str(args.lr_p) + '_' + str(args.batch_size) + '_' + str(args.width_p) + '_' + str(args.ns) + '.pth'
        # model_p.load_state_dict(torch.load('./weights/' + name, weights_only=True))
        # model_p.eval()
        # # test phase
        # for ii in range(10):
        #     test_data = generate_com_dataset(lamda, args.Wz, args.Wy, args.Nz, args.Ny, args.batch_size, args.K, args.Ns, args.ns, args.large_loss, Js)
        #     test_data = torch.from_numpy(test_data).float().to(device)
        #     x = model_p(test_data)
        #     model_p.zero_grad()
        #     model_d.zero_grad()
        #
        #     Rk = cal_obj(test_data, x, args, device).detach().cpu().numpy()
        #     print(np.min(Rk))
