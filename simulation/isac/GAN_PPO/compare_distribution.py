import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,  # 全局字体大小
    'axes.titlesize': 12,  # 标题字体大小
    'axes.labelsize': 12,  # x/y轴标签大小
    'xtick.labelsize': 12,  # x轴刻度标签大小
    'ytick.labelsize': 12,  # y轴刻度标签大小
    'legend.fontsize': 10  # 图例字体大小
})

matplotlib.use('Agg')
bins = np.linspace(-1, 1, 3)
print(bins)
indexes = np.digitize(-0.5, bins) - 1
print(indexes)


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



TD3 = np.load('../benchmark/TD3/TD3_mentocarlo.npy')
DDPG = np.load('../benchmark/TD3/DDPG_mentocarlo.npy')
DQN = np.load('../benchmark/DQN/VanillaDQN_mentocarlo.npy')
GPPO = np.load('GANPPO_mentocarlo.npy')
PPO = np.load('PPO_mentocarlo.npy')

# 随机生成数据（3组，每组100个样本）
# np.random.seed(42)
# data = {
#     'Group': np.repeat(['A', 'B', 'C'], 100),  # 组别标签
#     'Value': np.concatenate([
#         np.random.normal(0, 1, 100),    # 组A：均值0，标准差1
#         np.random.normal(5, 1.5, 100),  # 组B：均值5，标准差1.5
#         np.random.normal(10, 2, 100)    # 组C：均值10，标准差2
#     ])
# }
data = {
    'Algorithm': np.repeat(['PD-GPPO','PPO','TD3','DDPG','DQN'], 100),  # 组别标签
    'Maximum achievable rate (bps/Hz)': np.concatenate([GPPO, PPO, TD3, DDPG, DQN])
}
df = pd.DataFrame(data)

# 绘制基础小提琴图
plt.figure(figsize=(8, 6))
ax = sns.violinplot(x='Algorithm', y='Maximum achievable rate (bps/Hz)', hue='Algorithm',
               data=df, palette=['skyblue', 'm', 'cyan', 'lightgreen', 'slategray'])
# # 计算并标注中位数
# for i, day in enumerate(df["Algorithm"].unique()):
#     day_data = df[df["Algorithm"] == day]["Optimal objective function value"]
#     median = np.median(day_data)
#
#     # 在图中标注中位数数值
#     ax.text(i, median, f'{median:.3f}',
#             ha='center', va='center',
#             color='white', fontweight='bold',
#             bbox=dict(facecolor='black', alpha=0.8, boxstyle='round'))

plt.grid()
# plt.ylim([2, 7.5])
plt.savefig('./figs/violin_isac.eps',dpi=1600)
print(np.median(GPPO), np.median(PPO))