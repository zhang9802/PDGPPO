import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,  # 全局字体大小
    'axes.titlesize': 10,  # 标题字体大小
    'axes.labelsize': 10,  # x/y轴标签大小
    'xtick.labelsize': 10,  # x轴刻度标签大小
    'ytick.labelsize': 10,  # y轴刻度标签大小
    'legend.fontsize': 10  # 图例字体大小
})


conv_32 = np.load('convergence_lr_0.0005_width_p_32.npy')
conv_64 = np.load('convergence_lr_0.0005_width_p_64.npy')
conv_128 = np.load('convergence_lr_0.0005_width_p_128.npy')
plt.figure()
plt.plot(conv_32,'r-.', label='width=32')
plt.plot(conv_64,'b--', label='width=64')
plt.plot(conv_128,'k-', label='width=128')
plt.xlabel('Batch index')
plt.ylabel('P-Net loss')
plt.legend()
# plt.show(dpi=1200)
plt.savefig('pd_sensing_con_width.eps',dpi=1200)