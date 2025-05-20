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
        self.linearIn = nn.Linear(self.args.Ns * (self.args.K + self.args.J) * 2, self.args.width_p) # input layer
        self.batchNorm = nn.BatchNorm1d(self.args.Ns * self.args.K * 2)
        self.linear = nn.ModuleList()
        for _ in range(self.args.depth_p):
            self.linear.append(nn.Linear(self.args.width_p, self.args.width_p)) # hidden layers

        self.linearOut = nn.Linear(self.args.width_p, self.args.Ns *2) # output layer

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
        x = x / power * np.sqrt(self.args.Pbs)
        return x

class ISAC(object):
    def __init__(self,
                 args,
                 Js):
                        
        self.Ns = args.Ns
        self.ns = args.ns
        self.J = args.J
        self.K = args.K
        self.large_loss = args.large_loss
        self.Pbs = args.Pbs
        self.sigma_n = args.sigma_n
        self.count = 0 # count variable
        
        self.action_dim = math.comb(self.Ns, self.ns)
        self.state_dim = 2 * self.Ns * (self.J + self.K +1)
        self.combinations = args.combinations
        self.model = ResNet(args).to("cpu")
        self.model.load_state_dict(torch.load('../PD_NET/weights/my_network_0.0005_128_128.pth', weights_only=True)) #load weights
        self.model.eval()

        # self.episode_t = 0
        dz = args.Wz / (args.Nz - 1)
        dy = args.Wy / (args.Ny - 1)

        self.h_t, theta, phi = generate_isac_dataset(args.lamda, args.Wz, args.Wy, args.Nz, args.Ny,
                                                     args.K, args.J, args.Ns, args.ns, args.large_loss, Js)
        np.savez('channels.npz',theta=theta,phi=phi, h_t=self.h_t) 
    def reset(self):
        # self.episode_t = 0
        # gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        # self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model
        indexes = choice(self.combinations)  #random selection matrix
        # indexes = self.combinations[0]  #random selection matrix
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        h_t_E = self.h_t @ E
        w = 1 / np.sqrt(2) * (np.random.randn(self.Ns, 1) + 1j*np.random.randn(self.Ns, 1))
        # w = np.ones((self.Ns, 1), dtype=complex)
        w = w / np.linalg.norm(w,'fro')
        w_E = E @ w
        self.state = np.hstack((np.real(h_t_E.reshape(1, -1)), np.imag(h_t_E.reshape(1, -1)), np.real(w.reshape(1, -1)), np.imag(w.reshape(1, -1))))[0,:]
        
        return self.state

    def _compute_reward(self, h_t_E, x):
        reward = 0

        for k in range(self.K):
            H_temp = np.vstack((np.expand_dims(np.real(h_t_E[k]).reshape(-1, 1), axis=0),
                                np.expand_dims(np.imag(h_t_E[k]).reshape(-1, 1), axis=0)))

            x_temp = np.vstack((np.expand_dims(np.real(x), axis=0), np.expand_dims(np.imag(x), axis=0)))
            reward = reward + np.log2(1 + np.linalg.norm(cmul_np(conjT_np(H_temp), x_temp), 'fro') ** 2 / self.sigma_n)

        return reward  #10


    def step(self, action):

        indexes = self.combinations[action]
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        h_t_E = self.h_t @ E
        input = np.hstack((np.real(h_t_E.reshape(1, -1)), np.imag(h_t_E.reshape(1, -1))))
        w = self.model(torch.from_numpy(input).float()).detach().numpy()   #todo, resnet 保存权重等
        w = (w[:,:self.Ns] + 1j * w[:,self.Ns:]).reshape(-1,1)
        w_E = E @ w
        reward = self._compute_reward(h_t_E, w)


        self.state = np.hstack((np.real(h_t_E.reshape(1, -1)), np.imag(h_t_E.reshape(1, -1)), np.real(w.reshape(1, -1)), np.imag(w.reshape(1, -1))))[0,:]

        done = False #todo
        # self.count += 1
        return self.state, reward, done, None

    def close(self):
        pass
