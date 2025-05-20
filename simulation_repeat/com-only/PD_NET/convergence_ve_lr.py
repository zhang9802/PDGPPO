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


conv_lr_1 = np.load('convergence_lr_0.0001.npy')
conv_lr_5 = np.load('convergence_lr_0.0005.npy')

plt.figure()
plt.plot(conv_lr_1,'r-.', label='lr=1e-4')
plt.plot(conv_lr_5,'b--', label='lr=5e-4')
plt.xlabel('Batch index')
plt.ylabel('P-Net loss')
plt.legend()
# plt.show(dpi=1200)
plt.savefig('pd_com_con_lr.eps',dpi=1200)