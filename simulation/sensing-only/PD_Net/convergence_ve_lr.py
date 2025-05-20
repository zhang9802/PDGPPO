import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,  # 全局字体大小
    'axes.titlesize': 16,  # 标题字体大小
    'axes.labelsize': 15,  # x/y轴标签大小
    'xtick.labelsize': 12,  # x轴刻度标签大小
    'ytick.labelsize': 12,  # y轴刻度标签大小
    'legend.fontsize': 13  # 图例字体大小
})

# conv_lr_0 = np.load('convergence_lr_0.001.npy')
conv_lr_1 = np.load('convergence_lr_0.0001.npy')
conv_lr_5 = np.load('convergence_lr_0.0005.npy')

plt.figure()
# plt.plot(conv_lr_0,'k-', label='lr=0.001')
plt.plot(conv_lr_5, color='blue',linestyle='-.', label='lr=0.0005')
plt.plot(conv_lr_1,color='darkviolet',linestyle='-',label='lr=0.0001')

plt.xlabel('Batch index')
plt.ylabel('P-Net loss')
plt.legend()
# plt.show(dpi=1200)
plt.savefig('pd_sen_con_lr.eps',dpi=1200)