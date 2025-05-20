import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
plt.rcParams.update({
    'font.size': 12,  # 全局字体大小
    'axes.titlesize': 12,  # 标题字体大小
    'axes.labelsize': 12,  # x/y轴标签大小
    'xtick.labelsize': 12,  # x轴刻度标签大小
    'ytick.labelsize': 12,  # y轴刻度标签大小
    'legend.fontsize': 10  # 图例字体大小
})
PPO = np.load('PPO_reward_vs_ns.npy')
GPPO = np.load('GANPPO_reward_vs_ns.npy')
TD3 = np.load('../benchmark/TD3/TD3_reward_vs_ns.npy')
DDPG = np.load('../benchmark/TD3/DDPG_reward_vs_ns.npy')
DQN = np.load('../benchmark/DQN/VanillaDQN_reward_vs_ns.npy')
ns = [3,4,5,6]
plt.figure()
plt.plot(ns,GPPO,color='paleturquoise', linestyle='-', marker='s',label='PD-GPPO')
plt.plot(ns,PPO,color='cornflowerblue', linestyle='-.', marker='v',label='PD-PPO')
plt.plot(ns,TD3,color='orchid', linestyle='--', marker='p',label='TD3')
plt.plot(ns,DDPG,'cyan',linestyle='-', marker='d',label='DDPG')
plt.plot(ns,DQN,'palegreen',linestyle='-.', marker='*',label='DQN')
plt.xlabel('Number of activated ports')
plt.ylabel('Sum of sensing power (W)')
plt.legend(loc='best')
plt.grid()
plt.xlim([3,6])
# plt.ylim([37,46])
plt.savefig('./figs/compare_sensing_reward_vs_ns.eps',dpi=900)
 
