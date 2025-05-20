import numpy as np
import math
from random import choice

from utils import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
# todo

class Sensing(object):
    def __init__(self,
                 args):
                        
        self.Ns = args.Ns
        self.ns = args.ns
        self.J = args.J
        self.sigma_n = args.sigma_n
        self.sigma_t = args.sigma_t
        # self.Js = Js
        self.large_loss = args.large_loss
        self.Pbs = args.Pbs
        self.Gam = args.Gam
        self.count = 0 # count variable
        
        self.action_dim = 2 * self.Ns + 1
        self.state_dim = 2 * self.Ns * (self.J + 1)
        self.combinations = args.combinations
        dz = args.Wz / (args.Nz - 1)
        dy = args.Wy / (args.Ny - 1)
        theta = (np.random.rand(self.J) - 0.5) * np.pi
        phi = (np.random.rand(self.J) - 0.5) * np.pi
        print(theta,phi)
        self.Ht = np.zeros((self.Ns, self.J), dtype=complex)
        for jj in range(self.J):
            self.Ht[:,jj] = np.sqrt(args.large_loss) * two_dim_steering(dy, dz, theta[jj], phi[jj], args.Ny, args.Nz, args.wave_length).reshape(args.Ns,)

        # self.episode_t = 0
        # self.gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))

    def reset(self):
        # self.episode_t = 0
        # gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        # self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model
        indexes = choice(self.combinations)  #random selection matrix
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        Ht_E = E @ self.Ht
        w = 1 / np.sqrt(2) * (np.random.randn(self.Ns, 1) + 1j*np.random.randn(self.Ns, 1))
        # w = np.ones((self.Ns, 1), dtype=complex)
        w = w / np.linalg.norm(w,'fro')
        w_E = E @ w
        self.state = np.hstack((np.real(self.Ht.reshape(1, -1)), np.imag(self.Ht.reshape(1, -1)), np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]

        return self.state

    def _compute_reward(self, Ht_E, x, Gam):
        reward = 0

        for k in range(self.J):
            H_temp = np.vstack((np.expand_dims(np.real(Ht_E[:,k]).reshape(-1, 1), axis=0),
                                np.expand_dims(np.imag(Ht_E[:,k]).reshape(-1, 1), axis=0)))

            x_temp = np.vstack((np.expand_dims(np.real(x), axis=0), np.expand_dims(np.imag(x), axis=0)))
            reward = reward + np.linalg.norm(cmul_np(conjT_np(H_temp), x_temp), 'fro')-Gam 
            # reward = reward + np.linalg.norm(cmul_np(conjT_np(H_temp), x_temp), 'fro')
        return reward * 50  #10


    def step(self, action):
        # print(action)
        # self.episode_t += 1
        bins = np.linspace(-1, 1, len(self.combinations))
        indexes = self.combinations[np.digitize(action[0,-1], bins) - 1]
        E = selection_matrix(indexes, self.Ns)  #random selection matrix
        Ht_E = E @ self.Ht

        w = (action[:,:self.Ns] + 1j * action[:,self.Ns:2*self.Ns]).reshape(-1,1)
        w = w / np.linalg.norm(w,'fro')  * np.sqrt(self.Pbs)
        w_E = E @ w
        reward = self._compute_reward(Ht_E, w, self.Gam)

        # gk = 1 / np.sqrt(2) * (np.random.randn(self.Ns, self.K) + 1j * np.random.randn(self.Ns, self.K))
        # self.Hk = np.sqrt(self.large_loss) * self.gk.conj().T @ self.Js   #random downlink channels following Jake's model

        self.state = np.hstack((np.real(self.Ht.reshape(1, -1)), np.imag(self.Ht.reshape(1, -1)), np.real(w_E.reshape(1, -1)), np.imag(w_E.reshape(1, -1))))[0,:]

        done = False #todo
        # self.count += 1
        return self.state, reward, w, done, None


    def close(self):
        pass