import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 5,  # 全局字体大小
    'axes.titlesize': 5,  # 标题字体大小
    'axes.labelsize': 5,  # x/y轴标签大小
    'xtick.labelsize': 4,  # x轴刻度标签大小
    'ytick.labelsize': 4,  # y轴刻度标签大小
    'legend.fontsize': 5  # 图例字体大小
})

GANPP0 = np.load('./results/GANPPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.8.npz')
PP0 = np.load('./results/PPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.8.npz')
TD3 = np.load('../benchmark/TD3/Learning Curves/TD3/15_1e-05_1e-05_episode_2_TD3.npz')
DDPG = np.load('../benchmark/TD3/Learning Curves/DDPG/15_1e-05_1e-05_episode_2_DDPG.npz')
DQN = np.load('../benchmark/DQN/results/VanillaDQN_lr_1e-05_hid_128_gamma_0.8_bs_128.npz')


fig, axes = plt.subplots(1, 5, figsize=(9, 2))  # (宽度, 高度) 

 

axes[0].plot(GANPP0['max_eps_list'], 'b-',linewidth=0.5)
axes[0].set_xlabel('Episodes')
axes[0].set_ylabel('Maximum objective function value')
axes[0].set_title('PD-GPPO')
 

axes[1].plot(PP0['max_eps_list'], 'b-',linewidth=0.5)
axes[1].set_xlabel('Episodes')
axes[1].set_ylabel('Maximum objective function value')
axes[1].set_title('PD-PPO')


axes[2].plot(TD3['max_eps_list'], 'b-',linewidth=0.5)
axes[2].set_xlabel('Episodes')
axes[2].set_ylabel('Maximum objective function value')
axes[2].set_title('TD3')


axes[3].plot(DDPG['max_eps_list'], 'b-',linewidth=0.5)
axes[3].set_xlabel('Episodes')
axes[3].set_ylabel('Maximum objective function value')
axes[3].set_title('DDPG')

axes[4].plot(DQN['max_eps_list'], 'b-',linewidth=0.5)
axes[4].set_xlabel('Episodes')
axes[4].set_ylabel('Maximum objective function value')
axes[4].set_title('PD-DQN')
  
plt.tight_layout()
plt.savefig('./figs/evolution_com.png',dpi=1600)