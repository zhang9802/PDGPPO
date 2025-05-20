import numpy as np
import torch
from scipy.special import jv
from scipy.linalg import sqrtm
from random import choice
import matplotlib.pyplot as plt
from scipy import io as sio
import torch.nn as nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from utils import two_dim_steering, selection_matrix
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
class ResNet(torch.nn.Module):
    def __init__(self, Ns, K, J, width_p, depth_p, Pbs):
        super(ResNet, self).__init__()
        self.Pbs = Pbs
        self.linearIn = nn.Linear(Ns * (K + J) * 2, width_p) # input layer
        self.batchNorm = nn.BatchNorm1d(Ns*K * 2)
        self.linear = nn.ModuleList()
        for _ in range(depth_p):
            self.linear.append(nn.Linear(width_p, width_p)) # hidden layers

        self.linearOut = nn.Linear(width_p, Ns *2) # output layer

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
        x = x / power * np.sqrt(self.Pbs)
        return x

plt.rcParams.update({
    'font.size': 8,  # 全局字体大小
    'axes.titlesize': 8,  # 标题字体大小
    'axes.labelsize': 8,  # x/y轴标签大小
    'xtick.labelsize': 8,  # x轴刻度标签大小
    'ytick.labelsize': 8,  # y轴刻度标签大小
    'legend.fontsize': 8  # 图例字体大小
})
GANPP0 = np.load('./results/GANPPO_la_1e-05_lc_8e-05_hid_128_eps_0.4_gamma_0.8.npz')
PP0 = np.load('./results/PPO_la_1e-05_lc_8e-05_hid_128_eps_0.4_gamma_0.8.npz')
TD3 = np.load('../benchmark/TD3/Learning Curves/TD3/15_1e-05_1e-05_episode_2_TD3.npz')
DDPG = np.load('../benchmark/TD3/Learning Curves/DDPG/15_1e-05_1e-05_episode_2_DDPG.npz')
DQN = np.load('../benchmark/DQN/results/VanillaDQN_lr_1e-05_hid_128_gamma_0.8_bs_128.npz')
channels = np.load('channels.npz')
theta_target = channels['theta']
phi_target = channels['phi']
print(theta_target, phi_target)



# def plot_2D_beampattern(Wy, Wz, theta_target, phi_target, Ny, Nz, lamda, w, E):
Wy = 0.1
Wz = 0.1
Ny, Nz = 6, 6

lamda = 3e8 / 3.4e9
Ns = 36
J = 2
K = 2
dz = Wz / (Nz - 1)
dy = Wy / (Ny - 1)
large_loss = 1e-3
Pbs = 1.0
width_p, depth_p = 128, 4
channels = np.load('channels.npz')
h_t = channels['h_t']


model = ResNet(Ns, K, J, width_p, depth_p, Pbs).to("cpu")
model.load_state_dict(torch.load('../PD_NET/weights/my_network_0.0005_128_128.pth', weights_only=True)) #load weights
model.eval()

E_GPPO = selection_matrix(GANPP0['max_index'], Ns)
E_PPO = selection_matrix(PP0['max_index'], Ns)
E_TD3 = selection_matrix(TD3['max_index'], Ns)
E_DDPG = selection_matrix(DDPG['max_index'], Ns)
E_DQN = selection_matrix(DQN['max_index'], Ns)


dz = Wz / (Nz - 1)
dy = Wy / (Ny - 1)
ny = np.arange(0, Ny, 1).reshape(-1, 1)
nz = np.arange(0, Nz, 1).reshape(-1, 1)
 
theta_all = np.linspace(-np.pi/2, np.pi/2, 100)
phi_all = np.linspace(-np.pi/2, np.pi/2, 100)
Y, Z = np.meshgrid(phi_all,theta_all)

h_t_GPPO =  h_t @ E_GPPO
input = np.hstack((np.real(h_t_GPPO.reshape(1, -1)), np.imag(h_t_GPPO.reshape(1, -1))))
w_GPPO = model(torch.from_numpy(input).float()).detach().numpy()   #todo, resnet 保存权重等
w_GPPO = (w_GPPO[:,:Ns] + 1j * w_GPPO[:,Ns:]).reshape(-1,1)

h_t_PPO = h_t @ E_PPO
input = np.hstack((np.real(h_t_PPO.reshape(1, -1)), np.imag(h_t_PPO.reshape(1, -1))))
w_PPO = model(torch.from_numpy(input).float()).detach().numpy()   #todo, resnet 保存权重等
w_PPO = (w_PPO[:,:Ns] + 1j * w_PPO[:,Ns:]).reshape(-1,1)

h_t_DQN = h_t @ E_DQN
input = np.hstack((np.real(h_t_DQN.reshape(1, -1)), np.imag(h_t_DQN.reshape(1, -1))))
w_DQN = model(torch.from_numpy(input).float()).detach().numpy()   #todo, resnet 保存权重等
w_DQN = (w_DQN[:,:Ns] + 1j * w_DQN[:,Ns:]).reshape(-1,1)
w_TD3 = TD3['w_opt']
w_DDPG = DDPG['w_opt']

P_GPPO = np.zeros((len(phi_all),len(theta_all)))
P_PPO = np.zeros((len(phi_all),len(theta_all)))
P_TD3 = np.zeros((len(phi_all),len(theta_all)))
P_DDPG = np.zeros((len(phi_all),len(theta_all)))
P_DQN = np.zeros((len(phi_all),len(theta_all)))
# steering_target = np.sqrt(large_loss)*two_dim_steering(dy, dz, channels['theta'], channels['phi'], Ny, Nz, lamda)
# target_power = []
# target_power.append(np.linalg.norm((E_GPPO @steering_target).conj().T @ w_GPPO,'fro'))
# target_power.append(np.linalg.norm((E_PPO @steering_target).conj().T @ w_PPO,'fro'))
# target_power.append(np.linalg.norm((E_DQN @steering_target).conj().T @ w_DQN,'fro'))
# target_power.append(np.linalg.norm((E_TD3 @steering_target).conj().T @ w_TD3,'fro'))
# target_power.append(np.linalg.norm((E_DDPG @steering_target).conj().T @ w_DDPG,'fro'))


for ii in range(len(theta_all)):
    for jj in range(len(phi_all)):
        theta_temp = theta_all[ii]
        phi_temp = phi_all[jj]
        steering_temp = E_GPPO @ two_dim_steering(dy, dz, theta_temp, phi_temp, Ny, Nz, lamda)
        P_GPPO[ii,jj] = np.linalg.norm(np.sqrt(large_loss)*steering_temp.conj().T @ w_GPPO,'fro')

        steering_temp = E_PPO @ two_dim_steering(dy, dz, theta_temp, phi_temp, Ny, Nz, lamda)
        P_PPO[ii,jj] = np.linalg.norm(np.sqrt(large_loss)*steering_temp.conj().T @ w_PPO,'fro')

        steering_temp = E_DQN @ two_dim_steering(dy, dz, theta_temp, phi_temp, Ny, Nz, lamda)
        P_DQN[ii,jj] = np.linalg.norm(np.sqrt(large_loss)*steering_temp.conj().T @ w_DQN,'fro')

        steering_temp = E_TD3 @ two_dim_steering(dy, dz, theta_temp, phi_temp, Ny, Nz, lamda)
        P_TD3[ii,jj] = np.linalg.norm(np.sqrt(large_loss)*steering_temp.conj().T @ w_TD3,'fro')

        steering_temp = E_DDPG @ two_dim_steering(dy, dz, theta_temp, phi_temp, Ny, Nz, lamda)
        P_DDPG[ii,jj] = np.linalg.norm(np.sqrt(large_loss)*steering_temp.conj().T @ w_DDPG,'fro')



fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 
mesh = plt.pcolormesh(Y, Z,  P_GPPO, cmap='viridis', shading='auto')  # 使用pcolormesh
cbar = fig.colorbar(mesh)
# cbar.set_label('Gain')  # 设置 colorbar 标签
# contour = ax2.contourf(Y, Z,  P_dB, levels=20, cmap='viridis')
# fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=5)
plt.scatter(phi_target, theta_target, s=10, c='white')
plt.ylabel('Azimuth angle')
plt.xlabel('Elevation angle')
plt.title('PD-GPPO')
plt.tight_layout()
plt.savefig('./figs/beampattern_isac_1.eps', dpi=600, bbox_inches='tight')


fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 
mesh = plt.pcolormesh(Y, Z,  P_PPO, cmap='viridis', shading='auto')  # 使用pcolormesh
cbar = fig.colorbar(mesh)
# cbar.set_label('Gain')  # 设置 colorbar 标签
# contour = ax2.contourf(Y, Z,  P_dB, levels=20, cmap='viridis')
# fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=5)
plt.scatter(phi_target, theta_target, s=10, c='white')
plt.ylabel('Azimuth angle')
plt.xlabel('Elevation angle')
plt.title('PD-PPO')
plt.tight_layout()
plt.savefig('./figs/beampattern_isac_2.eps', dpi=600, bbox_inches='tight')

fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 
mesh = plt.pcolormesh(Y, Z,  P_TD3, cmap='viridis', shading='auto')  # 使用pcolormesh
cbar = fig.colorbar(mesh)
# cbar.set_label('Gain')  # 设置 colorbar 标签
# contour = ax2.contourf(Y, Z,  P_dB, levels=20, cmap='viridis')
# fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=5)
plt.scatter(phi_target, theta_target, s=10, c='white')
plt.ylabel('Azimuth angle')
plt.xlabel('Elevation angle')
plt.title('TD3')
plt.tight_layout()
plt.savefig('./figs/beampattern_isac_3.eps', dpi=600, bbox_inches='tight')


fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 
mesh = plt.pcolormesh(Y, Z,  P_DDPG, cmap='viridis', shading='auto')  # 使用pcolormesh
cbar = fig.colorbar(mesh)
# cbar.set_label('Gain')  # 设置 colorbar 标签
# contour = ax2.contourf(Y, Z,  P_dB, levels=20, cmap='viridis')
# fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=5)
plt.scatter(phi_target, theta_target, s=10, c='white')
plt.ylabel('Azimuth angle')
plt.xlabel('Elevation angle')
plt.title('DDPG')
plt.tight_layout()
plt.savefig('./figs/beampattern_isac_4.eps', dpi=600, bbox_inches='tight')


fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 
mesh = plt.pcolormesh(Y, Z,  P_DQN, cmap='viridis', shading='auto')  # 使用pcolormesh
cbar = fig.colorbar(mesh)
# cbar.set_label('Gain')  # 设置 colorbar 标签
# contour = ax2.contourf(Y, Z,  P_dB, levels=20, cmap='viridis')
# fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=5)
plt.scatter(phi_target, theta_target, s=10, c='white')
plt.ylabel('Azimuth angle')
plt.xlabel('Elevation angle')
plt.title('PD-DQN')
plt.tight_layout()
plt.savefig('./figs/beampattern_isac_5.eps', dpi=600, bbox_inches='tight')