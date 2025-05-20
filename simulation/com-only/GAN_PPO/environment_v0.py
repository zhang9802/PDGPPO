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
        self.params = args
        self.linearIn = nn.Linear(self.params.Ns * self.params.K * 2, self.params.width_p) # input layer
        self.batchNorm = nn.BatchNorm1d(self.params.Ns * self.params.K * 2)
        self.linear = nn.ModuleList()
        for _ in range(self.params.depth_p):
            self.linear.append(nn.Linear(self.params.width_p, self.params.width_p)) # hidden layers

        self.linearOut = nn.Linear(self.params.width_p, self.params.Ns *2) # output layer

    def forward(self, x):

        # x = self.batchNorm(x)
        x = self.linearIn(x)  # input layer
        for layer in self.linear:
            x_temp = F.relu(layer(x)) # activation $\tanh^3$
            x = x_temp + x # residual connection

        x = self.linearOut(x) # output layer
        power = torch.linalg.norm(x, axis=1, keepdims=True).repeat(1, x.shape[1])
        # x = x / power * np.sqrt(self.params.Pbs)
        x = x / power * np.sqrt(self.params.Pbs)
        return x

class COM(object):
    def __init__(self,
                 args,
                 Js):
                        
        self.Ns = args.Ns
        self.ns = args.ns
        self.K = args.K
        self.sigma_n = args.sigma_n
        self.sigma_t = args.sigma_t
        self.Js = Js
        self.large_loss = args.large_loss
        self.Pbs = args.Pbs

        self.count = 0 # count variable
        
        self.action_dim = math.comb(self.Ns, self.ns)
        self.state_dim = 2 * self.Ns * (self.K + 1)
        self.combinations = args.combinations
        self.model = ResNet(args).to("cpu")
        weight_name='../PD_NET/weights/my_network_0.0005_128_128.pth'
        self.model.load_state_dict(torch.load(weight_name, weights_only=True)) #load weights
        self.model.eval()

        # self.episode_t = 0
        self.gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))

    def reset(self):
        # self.episode_t = 0
        # self.gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model
        # indexes = choice(self.combinations)  #random selection matrix
        indexes = self.combinations[0]  # random selection matrix
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        # Hk_E = self.Hk @ E
        w = 1 / np.sqrt(2) * (np.random.randn(self.Ns, 1) + 1j*np.random.randn(self.Ns, 1))
        # w = np.ones((self.Ns, 1), dtype=complex)
        w = w / np.linalg.norm(w,'fro') * np.sqrt(self.Pbs)
        w_E = E @ w * 1e1
        self.state = np.hstack((np.real(self.Hk.reshape(1, -1))/1e1, np.imag(self.Hk.reshape(1, -1))/1e1, np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]

        return self.state

    def _compute_reward(self, Hk_E, x):
        reward = 0

        for k in range(self.K):
            H_temp = np.vstack((np.expand_dims(np.real(Hk_E[k]).reshape(-1, 1), axis=0),
                                np.expand_dims(np.imag(Hk_E[k]).reshape(-1, 1), axis=0)))

            x_temp = np.vstack((np.expand_dims(np.real(x), axis=0), np.expand_dims(np.imag(x), axis=0)))
            reward = reward + np.log2(1 + np.linalg.norm(cmul_np(conjT_np(H_temp), x_temp), 'fro') ** 2 / self.sigma_n)

        return reward  #10


    def step(self, action):
        # print(action)
        # self.episode_t += 1
        indexes = self.combinations[action]
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        Hk_E = self.Hk @ E
        input = np.hstack((np.real(Hk_E.reshape(1, -1)), np.imag(Hk_E.reshape(1, -1))))
        w = self.model(torch.from_numpy(input).float()).detach().numpy()   #todo, resnet 保存权重等
        w = (w[:,:self.Ns] + 1j * w[:,self.Ns:]).reshape(-1,1)
        w_E = E @ w * 1e1
        reward = self._compute_reward(Hk_E, w)   #todo

        # gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        # self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model

        self.state = np.hstack((np.real(self.Hk.reshape(1, -1))/1e1, np.imag(self.Hk.reshape(1, -1))/1e1, np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]

        done = False #todo
        # self.count += 1
        return self.state, reward, done, None

    def close(self):
        pass