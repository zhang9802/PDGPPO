import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
 
plt.rcParams.update({
    'font.size': 8,  # 全局字体大小
    'axes.titlesize': 8,  # 标题字体大小
    'axes.labelsize': 8,  # x/y轴标签大小
    'xtick.labelsize': 8,  # x轴刻度标签大小
    'ytick.labelsize': 8,  # y轴刻度标签大小
    'legend.fontsize': 8  # 图例字体大小
})

GANPP0 = np.load('./results/GANPPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.9.npz')
PP0 = np.load('./results/PPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.9.npz')
TD3 = np.load('../benchmark/TD3/Learning Curves/TD3/15_1e-05_1e-05_episode_2_TD3.npz')
DDPG = np.load('../benchmark/TD3/Learning Curves/DDPG/15_1e-05_1e-05_episode_2_DDPG.npz')
DQN = np.load('../benchmark/DQN/results/VanillaDQN_lr_1e-05_hid_128_gamma_0.9_bs_128.npz')


fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 

plt.plot(GANPP0['max_eps_list'], 'b-',linewidth=0.5)
plt.xlabel('Episodes')
plt.ylabel('Maximum objective function value')
plt.title('PD-GPPO')
plt.tight_layout()
plt.savefig('./figs/evolution_com_1.eps', dpi=600, bbox_inches='tight')
 

fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 

plt.plot(PP0['max_eps_list'], 'b-',linewidth=0.5)
plt.xlabel('Episodes')
plt.ylabel('Maximum objective function value')
plt.title('PD-PPO')
plt.tight_layout()
plt.savefig('./figs/evolution_com_2.eps', dpi=600, bbox_inches='tight')
 


fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 

plt.plot(TD3['max_eps_list'], 'b-',linewidth=0.5)
plt.xlabel('Episodes')
plt.ylabel('Maximum objective function value')
plt.title('TD3')
plt.tight_layout()
plt.savefig('./figs/evolution_com_3.eps', dpi=600, bbox_inches='tight')
 


fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 

plt.plot(DDPG['max_eps_list'], 'b-',linewidth=0.5)
plt.xlabel('Episodes')
plt.ylabel('Maximum objective function value')
plt.title('DDPG')
plt.tight_layout()
plt.savefig('./figs/evolution_com_4.eps', dpi=600, bbox_inches='tight')


fig=plt.figure(figsize=(3, 3))  # (宽度, 高度) 

plt.plot(DQN['max_eps_list'], 'b-',linewidth=0.5)
plt.xlabel('Episodes')
plt.ylabel('Maximum objective function value')
plt.title('PD-DQN')
plt.tight_layout()
plt.savefig('./figs/evolution_com_5.eps', dpi=600, bbox_inches='tight')

  

