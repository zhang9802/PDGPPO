import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import one2two
import matplotlib
matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__=="__main__":
    plt.rcParams.update({
        'font.size': 12,  # 全局字体大小
        'axes.titlesize': 12,  # 标题字体大小
        'axes.labelsize': 12,  # x/y轴标签大小
        'xtick.labelsize': 12,  # x轴刻度标签大小
        'ytick.labelsize': 12,  # y轴刻度标签大小
        'legend.fontsize': 10  # 图例字体大小
    })
    name1 = 'compare_sensing_conv_v1'
    GANPP0 = np.load('./results/GANPPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.8.npz')
    mv_GANPP0 = rl_utils.moving_average(GANPP0['return_list'], 9)
    PP0 = np.load('./results/PPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.8.npz')
    mv_PP0 = rl_utils.moving_average(PP0['return_list'], 9)

    TD3 = np.load('../benchmark/TD3/Learning Curves/TD3/15_1e-05_1e-05_episode_2_TD3.npz')
    mv_TD3 = rl_utils.moving_average(TD3['return_list'], 9)
    DDPG = np.load('../benchmark/TD3/Learning Curves/DDPG/15_1e-05_1e-05_episode_2_DDPG.npz')
    mv_DDPG = rl_utils.moving_average(DDPG['return_list'], 9)
    DQN = np.load('../benchmark/DQN/results/VanillaDQN_lr_1e-05_hid_128_gamma_0.8_bs_128.npz')
    mv_DQN = rl_utils.moving_average(DQN['return_list'], 9)
    plt.figure(figsize=(8, 6))
    plt.plot( GANPP0['return_list'],color='paleturquoise')
    plt.plot(mv_GANPP0,color='darkturquoise',label='PD-GPPO')
    plt.plot( PP0['return_list'],color='cornflowerblue')
    plt.plot(mv_PP0,color='mediumblue',label='PPO')
    plt.plot( TD3['return_list'],color='orchid')
    plt.plot(mv_TD3,color='darkviolet',label='TD3')
    plt.plot( DDPG['return_list'],'paleturquoise')
    plt.plot(mv_DDPG,'aqua',label='DDPG')
    plt.plot( DQN['return_list'],'palegreen')
    plt.plot(mv_DQN,'mediumspringgreen',label='PD-DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.xlim([0,1000])
    # plt.ylim([420,690])
    # plt.tight_layout()
    plt.legend(loc='center left')
    # plt.savefig(name1 + '.png', dpi=1600)
    plt.savefig('./figs/' + name1 + '.eps', dpi=1600)

    name2 = 'compare_sensing_pos_v1.eps'

    Ny = 6
    plt.figure(figsize=(6, 6))

    # 生成网格点
    x = np.arange(0, Ny, 1)
    y = np.arange(0, Ny, 1)
    X, Y = np.meshgrid(x, y)

    rows_GAN, cols_GAN = [], []
    for i in range(5):
        col_temp, row_temp = one2two(GANPP0['max_index'][i], Ny)
        rows_GAN.append(row_temp)
        cols_GAN.append(col_temp)

    rows_PPO, cols_PPO = [], []
    for i in range(5):
        col_temp, row_temp = one2two(PP0['max_index'][i], Ny)
        rows_PPO.append(row_temp)
        cols_PPO.append(col_temp)

    rows_TD3, cols_TD3 = [], []
    for i in range(5):
        col_temp, row_temp = one2two(TD3['max_index'][i], Ny)
        rows_TD3.append(row_temp)
        cols_TD3.append(col_temp)

    rows_DDPG, cols_DDPG = [], []
    for i in range(5):
        col_temp, row_temp = one2two(DDPG['max_index'][i], Ny)
        rows_DDPG.append(row_temp)
        cols_DDPG.append(col_temp)

    rows_DQN, cols_DQN = [], []
    for i in range(5):
        col_temp, row_temp = one2two(DQN['max_index'][i], Ny)
        rows_DQN.append(row_temp)
        cols_DQN.append(col_temp)

    # 绘制网格线
    plt.grid(True, linestyle='--',  linewidth=1)

    # 在交点处绘制小圆点（但不是 scatter）
    # plt.plot(cols, rows, 'o', color='blue', markersize=6)
    plt.scatter(
        cols_GAN, rows_GAN,
        s=120,  # 点的大小
        facecolors='none',  # 无填充
        edgecolors='darkturquoise',  # 边缘颜色
        linewidths=1.5,  # 边缘线宽
        marker='o',  # 圆形标记
        label='PD-GPPO'
    )
    plt.scatter(
        cols_PPO, rows_PPO,
        s=120,  # 点的大小
        facecolors='none',  # 无填充
        edgecolors='mediumblue',  # 边缘颜色
        linewidths=1.5,  # 边缘线宽
        marker='v',  # 圆形标记
        label='PD-PPO'
    )

    plt.scatter(
        cols_TD3, rows_TD3,
        s=120,  # 点的大小
        facecolors='none',  # 无填充
        edgecolors='darkviolet',  # 边缘颜色
        linewidths=1.5,  # 边缘线宽
        marker='d',  # 圆形标记
        label='TD3'
    )

    plt.scatter(
        cols_DDPG, rows_DDPG,
        s=120,  # 点的大小
        facecolors='none',  # 无填充
        edgecolors='aqua',  # 边缘颜色
        linewidths=1.5,  # 边缘线宽
        marker='*',  # 圆形标记
        label='DDPG'
    )

    plt.scatter(
        cols_DQN, rows_DQN,
        s=120,  # 点的大小
        facecolors='none',  # 无填充
        edgecolors='mediumspringgreen',  # 边缘颜色
        linewidths=1.5,  # 边缘线宽
        marker='p',  # 圆形标记
        label='PD-DQN'
    )

    plt.xlim(-0.5, Ny - 0.5)
    plt.ylim(-0.5, Ny - 0.5)
    plt.xlabel('y axis')
    plt.ylabel('z axis')
    plt.legend(loc='lower right')
    # plt.title("6×6 Grid (Small Markers)")
    plt.savefig('./figs/' + name2, dpi=1600)

    name3 = 'compare_sensing_val_v1.eps'

    categories = ['PD-GPPO', 'PD-PPO', 'TD3', 'DDPG', 'PD-DQN']
    values  = []
    values.append(GANPP0['max_reward'])
    values.append(PP0['max_reward'])
    values.append(TD3['max_reward'])
    values.append(DDPG['max_reward'])
    values.append(DQN['max_reward'])
    values = np.array(values)
    values = np.round(values, 3)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']  # 自定义颜色列表

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 创建柱状图
    bars = ax.bar(categories, values, color=colors, edgecolor='black', width=0.7, linewidth=1.2)

    # 在每个柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height}',
                ha='center', va='bottom',
                fontsize=14)

    # 设置坐标轴线样式
    ax.spines['bottom'].set_linewidth(2)  # X轴线加粗
    ax.spines['left'].set_linewidth(2)  # Y轴线加粗

    # 设置刻度样式
    ax.tick_params(axis='y', width=2, direction='in',labelsize=16)  # Y轴刻度朝内
    ax.tick_params(axis='x', width=2,labelsize=16,rotation=15,pad=5)  # X轴刻度加粗

    # 添加标题和标签
    # ax.set_title('改进版柱状图示例', fontsize=14, pad=20)
    ax.set_xlabel(' ', fontsize=12, labelpad=10)
    ax.set_ylabel('Maximum sensing power (W)', fontsize=16, labelpad=10)

    # 调整y轴范围
    ax.set_ylim(0, max(values) * 1.15)

    # 自定义网格和边框
    # ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('./figs/' + name3, dpi=1600)


