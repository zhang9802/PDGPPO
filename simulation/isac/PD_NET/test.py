import numpy as np
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

from scipy.special import jv
from random import choice


fc = 3.4 * 1e9
lamda = 3e8/fc    # waveform length
Wz = 0.1     # length along z-axis
Wy = 0.1     # length along y-axis
Nz = 8     # number of ports along Wy
Ny = 8     # number of ports along Wy
Ns = Ny * Nz
ns = 5  # number of activated ports
Pmax = 0.1  # transmit  power at BS, W
K = 2   #number of users
large_loss = 1e-3 #large-scale fading
hk = JakeModel(lamda, Wz, Wy, Nz, Ny, large_loss)
x, y = one2two(3, 4)

print(x), print(y)

b = list(range(10))


# 示例使用
numbers = [1, 2, 3, 4, 5]
m = 3
combinations = get_combinations_recursive(b, m)
print(combinations)
print(choice(combinations)) # 随机抽取一个
ind = [0,2,3]
print(selection_matrix(ind, 5))

a = np.arange(6).reshape((1,3,2))

b = a.reshape((1,-1))
print(b)
print(b.reshape(1,3,2))