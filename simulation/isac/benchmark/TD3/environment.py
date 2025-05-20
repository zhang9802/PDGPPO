import numpy as np
import math
from random import choice

from utils import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
# todo

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
        self.count = 0  # count variable

        self.action_dim = 2 * self.Ns + 1
        self.state_dim = 2 * self.Ns * (self.J + self.K + 1)
        self.combinations = args.combinations


        # self.episode_t = 0
        dz = args.Wz / (args.Nz - 1)
        dy = args.Wy / (args.Ny - 1)

        self.h_t, theta, phi = generate_isac_dataset(args.lamda, args.Wz, args.Wy, args.Nz, args.Ny,
                                                     args.K, args.J, args.Ns, args.ns, args.large_loss, Js)


    def reset(self):

        indexes = self.combinations[0]  # random selection matrix
        E = selection_matrix(indexes, self.Ns)  # random selection matrix
        h_t_E = self.h_t @ E
        w = 1 / np.sqrt(2) * (np.random.randn(self.Ns, 1) + 1j * np.random.randn(self.Ns, 1))

        w = w / np.linalg.norm(w, 'fro')
        w_E = E @ w
        self.state = np.hstack((np.real(h_t_E.reshape(1, -1)), np.imag(h_t_E.reshape(1, -1)), np.real(w.reshape(1, -1)),
                                np.imag(w.reshape(1, -1))))[0, :]

        return self.state

    def _compute_reward(self, h_t_E, x):
        reward = 0

        for k in range(self.K):
            H_temp = np.vstack((np.expand_dims(np.real(h_t_E[k]).reshape(-1, 1), axis=0),
                                np.expand_dims(np.imag(h_t_E[k]).reshape(-1, 1), axis=0)))

            x_temp = np.vstack((np.expand_dims(np.real(x), axis=0), np.expand_dims(np.imag(x), axis=0)))
            reward = reward + np.log2(1 + np.linalg.norm(cmul_np(conjT_np(H_temp), x_temp), 'fro') ** 2 / self.sigma_n)

        return reward  # 10


    def step(self, action):

        bins = np.linspace(-1, 1, len(self.combinations))
        indexes = self.combinations[np.digitize(action[0,-1], bins) - 1]
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        h_t_E = self.h_t @ E

        w = (action[:,:self.Ns] + 1j * action[:,self.Ns:2*self.Ns]).reshape(-1,1)
        w = w / np.linalg.norm(w,'fro') * np.sqrt(self.Pbs)
        w_E = E @ w
        reward = self._compute_reward(h_t_E, w)

        self.state = np.hstack((np.real(h_t_E.reshape(1, -1)), np.imag(h_t_E.reshape(1, -1)), np.real(w.reshape(1, -1)),np.imag(w.reshape(1, -1))))[0, :]

        done = False #todo

        return self.state, reward, w, done, None



