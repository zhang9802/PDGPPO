import numpy as np
import math
from random import choice

from utils import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
# todo

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
        
        self.action_dim = 2 * self.Ns + 1
        self.state_dim = 2 * self.Ns * (self.K + 1)
        self.combinations = args.combinations
        self.Gam = args.Gam

        # self.episode_t = 0
        self.gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))

    def reset(self):
        # self.episode_t = 0
        # gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model
        indexes = self.combinations[0]  #random selection matrix
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        # Hk_E = self.Hk @ E
        w = 1 / np.sqrt(2) * (np.random.randn(self.Ns, 1) + 1j*np.random.randn(self.Ns, 1))
        # w = np.ones((self.Ns, 1), dtype=complex)
        w = w / np.linalg.norm(w,'fro') * np.sqrt(self.Pbs)
        w_E = E @ w * 1e1
        self.state = np.hstack((np.real(self.Hk.reshape(1, -1))/1e1, np.imag(self.Hk.reshape(1, -1))/1e1, np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]

        return self.state

    def _compute_reward(self, Hk_E, x, Gam):
        reward = 0

        for k in range(self.K):
            H_temp = np.vstack((np.expand_dims(np.real(Hk_E[k]).reshape(-1, 1), axis=0),
                                np.expand_dims(np.imag(Hk_E[k]).reshape(-1, 1), axis=0)))

            x_temp = np.vstack((np.expand_dims(np.real(x), axis=0), np.expand_dims(np.imag(x), axis=0)))
            Rk_temp = np.log2(1 + np.linalg.norm(cmul_np(conjT_np(H_temp), x_temp), 'fro') ** 2 / self.sigma_n)
            reward = reward + Rk_temp + np.min([Rk_temp - Gam, 0])

        return reward   #10


    def step(self, action):
        # print(action)
        # self.episode_t += 1
        bins = np.linspace(-1, 1, len(self.combinations))
        indexes = self.combinations[np.digitize(action[0,-1], bins) - 1]
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        Hk_E = self.Hk @ E

        w = (action[:,:self.Ns] + 1j * action[:,self.Ns:2*self.Ns]).reshape(-1,1)
        w = w / np.linalg.norm(w,'fro') * np.sqrt(self.Pbs)
        w_E = E @ w * 1e1
        reward = self._compute_reward(Hk_E, w, self.Gam)

        # gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        # self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model

        self.state = np.hstack((np.real(self.Hk.reshape(1, -1))/1e1, np.imag(self.Hk.reshape(1, -1))/1e1, np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]

        done = False #todo
        # self.count += 1
        return self.state, reward, done, None



