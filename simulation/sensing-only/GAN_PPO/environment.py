import numpy as np
import math
from random import choice

from utils import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
# todo
class ResNet(torch.nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args
        self.linearIn = nn.Linear(args.Ns * 2 * args.J, args.width_p) # input layer
        self.batchNorm = nn.BatchNorm1d(args.Ns * 2)
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
        # constant modulus constraint
        # power1 = torch.sqrt(x[:,:args.Ns]**2 + x[:,args.Ns:]**2)
        # power = torch.cat((power1, power1), dim=1)
        x = x / power * np.sqrt(self.args.Pbs)
        return x

class Sensing(object):
    def __init__(self,
                 args):
                        
        self.Ns = args.Ns
        self.ns = args.ns
        self.J = args.J
        # self.sigma_n = args.sigma_n
        # self.sigma_t = args.sigma_t
        # self.Js = Js
        self.large_loss = args.large_loss
        self.Pbs = args.Pbs

        self.count = 0 # count variable
        
        self.action_dim = math.comb(self.Ns, self.ns)
        self.state_dim = 2 * self.Ns * (self.J + 1)
        self.combinations = args.combinations
        self.model = ResNet(args).to("cpu")
        weight_name='../PD_Net/weights/my_network_0.0001_128_128_' + str(args.ns) + '.pth'
        self.model.load_state_dict(torch.load(weight_name, weights_only=True)) #load weights
        self.model.eval()

        # self.episode_t = 0
        dz = args.Wz / (args.Nz - 1)
        dy = args.Wy / (args.Ny - 1)
        theta = (np.random.rand(self.J) - 0.5) * np.pi
        phi = (np.random.rand(self.J) - 0.5) * np.pi
        print(theta,phi)
        # np.savez('channels.npz',theta=theta,phi=phi) 
        self.h_t = np.zeros((self.Ns, self.J), dtype=complex)
        for jj in range(args.J):
            self.h_t[:,jj] =  np.sqrt(args.large_loss) * two_dim_steering(dy, dz, theta[jj], phi[jj], args.Ny, args.Nz, args.lamda).reshape(args.Ns,)

    def reset(self):
        # self.episode_t = 0
        # gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        # self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model
        indexes = choice(self.combinations) # random selection matrix  #random selection matrix
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        # h_t_E = E @ self.h_t
        w = 1 / np.sqrt(2) * (np.random.randn(self.Ns, 1) + 1j*np.random.randn(self.Ns, 1))
        # w = np.ones((self.Ns, 1), dtype=complex)
        w = w / np.linalg.norm(w,'fro') * np.sqrt(self.Pbs)
        w_E = E @ w
        self.state = np.hstack((np.real(self.h_t.reshape(1, -1)), np.imag(self.h_t.reshape(1, -1)), np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]
        
        return self.state

    def _compute_reward(self, h_t_E, x):
        reward = 0

        for k in range(self.J):
            H_temp = np.vstack((np.expand_dims(np.real(h_t_E[:,k]).reshape(-1, 1), axis=0),
                                np.expand_dims(np.imag(h_t_E[:,k]).reshape(-1, 1), axis=0)))

            x_temp = np.vstack((np.expand_dims(np.real(x), axis=0), np.expand_dims(np.imag(x), axis=0)))
            reward = reward + np.linalg.norm(cmul_np(conjT_np(H_temp), x_temp), 'fro')

        return reward *50 #10


    def step(self, action):
        # print(action)
        # self.episode_t += 1
        indexes = self.combinations[action]
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        h_t_E = E @ self.h_t
        input = np.hstack((np.real(h_t_E.reshape(1, -1)), np.imag(h_t_E.reshape(1, -1))))
        w = self.model(torch.from_numpy(input).float()).detach().numpy()   #todo, resnet 保存权重等
        w = (w[:,:self.Ns] + 1j * w[:,self.Ns:]).reshape(-1,1)

        reward = self._compute_reward(h_t_E, w)
        w_E = E @ w 
        
        self.state = np.hstack((np.real(self.h_t.reshape(1, -1)), np.imag(self.h_t.reshape(1, -1)), np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]


        done = False #todo
        # self.count += 1
        return self.state, reward, done, None

    def close(self):
        pass