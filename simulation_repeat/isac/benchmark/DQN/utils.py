import numpy as np
import torch
from scipy.special import jv
from scipy.linalg import sqrtm
from random import choice
import matplotlib.pyplot as plt
from scipy import io as sio

from matplotlib import cm
from matplotlib.ticker import LinearLocator
def cmul_np(A, B):
    # complex matrix multiplication C = A * B
    # only accept 2 dimension matrix mul or matrix mul a scalar or scalar mul a scalar
    # please do not input vectors, unsqueeze to n*1(1*n) matrix instead

    Cre = A[0, :] @ B[0, :] - A[1, :] @ B[1, :]
    Cim = A[0, :] @ B[1, :] + A[1, :] @ B[0, :]
    return np.hstack((Cre, Cim))


def conjT_np(A):
    # conjugate transpose of a matrix
    Bre = A[0, :].T
    Bim = -A[1, :].T
    return np.vstack((np.expand_dims(Bre,axis=0), np.expand_dims(Bim,axis=0)))


def cmul(A, B):
    # complex matrix multiplication C = A * B
    # only accept 2 dimension matrix mul or matrix mul a scalar or scalar mul a scalar
    # please do not input vectors, unsqueeze to n*1(1*n) matrix instead

    Cre = torch.mm(A[0, :], B[0, :]) - torch.mm(A[1, :], B[1, :])
    Cim = torch.mm(A[0, :], B[1, :]) + torch.mm(A[1, :], B[0, :])
    return torch.cat((Cre, Cim), 1)


def conjT(A):
    # conjugate transpose of a matrix
    if A.size(0) != 2:
        raise Exception("The first input tensor dimension must be 2")

    if A.dim() >= 2:
        Bre = A[0, :].T
        Bim = -A[1, :].T
        return torch.cat((Bre.unsqueeze(0), Bim.unsqueeze(0)), 0)

    # else:
    #     return conj(A)

def one2two(z, Ny):

    x = z % Ny
    y = np.floor(z / Ny)
    return x, y


def JakeModel(lamb, Wz, Wy, Nz, Ny, large_loss):
    """
    Jake model for 2D fluid antenna
    :param lamb:
    :param Wz:
    :param Wy:
    :param Nz:
    :param Ny:
    :param large_loss: large-scale fading
    :return:
    """
    Ns = Ny * Nz
    Js = np.zeros((Ns,Ns), dtype=complex)
    dis = np.zeros((Ns,Ns), dtype=float)

    for i in range(Ns):
        for j in range(Ns):
            yii, zii = one2two(i, Ny)
            yjj, zij = one2two(j, Nz)
            dis[i, j] = np.sqrt(((yii - yjj) / (Ny-1) * Wy)**2 + ((zii - zij) / (Nz-1) * Wz)**2)
            Js[i, j] = jv(0, 2 * np.pi * dis[i, j] / lamb)

    gk = 1 / np.sqrt(2) * (np.random.randn(Ns, 1) + 1j * np.random.randn(Ns, 1))
    U, S, V = np.linalg.svd(Js)
    hk = (np.sqrt(large_loss) * gk.conj().T @ np.sqrt(np.diag(S)) @ V).conj().T
    return hk

def JakeModelV1(lamb, Wz, Wy, Nz, Ny, large_loss):
    """
    Jake model for 2D fluid antenna
    :param lamb:
    :param Wz:
    :param Wy:
    :param Nz:
    :param Ny:
    :param large_loss: large-scale fading
    :return:
    """
    Ns = Ny * Nz
    Js = np.zeros((Ns,Ns), dtype=complex)
    dis = np.zeros((Ns,Ns), dtype=float)

    for i in range(Ns):
        for j in range(Ns):
            yii, zii = one2two(i, Ny)
            yjj, zij = one2two(j, Nz)
            dis[i, j] = np.sqrt(((yii - yjj) / (Ny-1) * Wy)**2 + ((zii - zij) / (Nz-1) * Wz)**2)
            Js[i, j] = jv(0, 2 * np.pi * dis[i, j] / lamb)

    U, S, V = np.linalg.svd(Js)
    return np.sqrt(np.diag(S)) @ V


def LosModel(lamb, Wz, Wy, Nz, Ny, theta, phi):
    """
    LOSS model for 2D fluid antenna
    :param lamb:
    :param Wz:
    :param Wy:
    :param Nz:
    :param Ny:
    :param theta:
    :param phi:
    :return:
    """
    dz = Wz / (Nz - 1)
    dy = Wy / (Ny - 1)
    ny = np.arange(0, Ny, 1).reshape(-1, 1)
    nz = np.arange(0, Nz, 1).reshape(-1, 1)
    steer_y = np.exp(1j* 2 * np.pi * ny * dy / lamb * np.cos(phi) * np.sin(theta)).reshape(-1, 1)
    steer_z = np.exp(1j* 2 * np.pi * nz * dz / lamb * np.sin(phi)).reshape(-1, 1)

    steer_vector = np.kron(steer_y, steer_z)

    return steer_vector

def get_combinations_recursive(numbers, m):
    """

    :param numbers: list of antenna indexes
    :param m: number of activated ports
    :return: all combinations
    """
    result = []
    if m == 0:
        return [[]]
    if len(numbers) < m:
        return []
    first = numbers[0]
    for smaller_combinations in get_combinations_recursive(numbers[1:], m - 1):
        new_combination = [first] + smaller_combinations
        result.append(new_combination)
    for smaller_combinations in get_combinations_recursive(numbers[1:], m):
        result.append(smaller_combinations)
    return result

def selection_matrix(indexes, Ns):
    """
    activated ports -> selection matrix
    :param indexes: indexes of activated ports
    :param Ns: number of ports
    :return: selection matrix
    """
    x = np.zeros(Ns, dtype=int)
    x[indexes] = 1
    E = np.diag(x)
    return E

def generate_sensing_dataset(lamda, Wz, Wy, Nz, Ny, n_samples, Ns, ns, large_loss, combinations, J):
    """
    generate com dataset
    :param lamda: wavelength
    :param Wz:
    :param Wy:
    :param Nz:
    :param Ny:
    :param n_samples:
    :param Ns:
    :param ns:
    :param large_loss:
    :return:
    """
    dz = Wz / (Nz - 1)
    dy = Wy / (Ny - 1)
    Hk = np.zeros((n_samples, Ns, J), dtype=complex)    # Hk @ E matrix
    theta_all = []
    phi_all = []
    indexes_all = []

    for iter in range(n_samples):
        index = choice(combinations)
        E = selection_matrix(index, Ns)
        theta = (np.random.rand(J) - 0.5) * np.pi
        phi = (np.random.rand(J) - 0.5) * np.pi
        theta_all.append(theta)
        phi_all.append(phi)
        indexes_all.append(index)
        for jj in range(J):
            Hk[iter,:,jj] = (E @ two_dim_steering(dy, dz, theta[jj], phi[jj], Ny, Nz, lamda)).reshape(Ns,)

    Hk_all = np.hstack((np.real(Hk).reshape(n_samples,-1), np.imag(Hk).reshape(n_samples,-1)))
    return Hk_all, theta_all, phi_all, indexes_all


def two_dim_steering(dy, dz, theta, phi, Ny, Nz, lamda):
    """

    :param dy:
    :param dz:
    :param theta: 俯仰角
    :param phi: 方位角
    :param Ny:
    :param Nz:
    :return:
    """
    ny = np.arange(0, Ny, 1).reshape(-1, 1)
    nz = np.arange(0, Nz, 1).reshape(-1, 1)

    Fy = np.exp(1j * ny * 2 * np.pi * np.sin(theta) * dy * np.cos(phi) / lamda).reshape(-1, 1)
    Fz = np.exp(1j * nz * 2 * np.pi * np.sin(phi)* dz / lamda).reshape(-1, 1)

    return np.kron(Fy, Fz)

def plot_2D_beampattern(Wy, Wz, theta_target, phi_target, Ny, Nz, lamda, w, E):

    dz = Wz / (Nz - 1)
    dy = Wy / (Ny - 1)
    ny = np.arange(0, Ny, 1).reshape(-1, 1)
    nz = np.arange(0, Nz, 1).reshape(-1, 1)

    theta_all = np.linspace(-np.pi/2, np.pi/2, 100)
    phi_all = np.linspace(-np.pi/2, np.pi/2, 100)
    Y, Z = np.meshgrid(phi_all,theta_all)
    P_all = np.zeros((len(phi_all),len( theta_all)))
    steering_target = two_dim_steering(dy, dz, theta_target, phi_target, Ny, Nz, lamda)
    for ii in range(len(theta_all)):
        for jj in range(len(phi_all)):
            theta_temp = theta_all[ii]
            phi_temp = phi_all[jj]
            steering_temp = E @ two_dim_steering(dy, dz, theta_temp, phi_temp, Ny, Nz, lamda)
            P_all[ii,jj] = np.linalg.norm(steering_temp.conj().T @ w,'fro')**2

    # P_dB = 10 * np.log10(P_all/np.max(np.max(P_all)))
    P_dB = P_all
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    # Plot the surface.
    surf = ax.plot_surface(Y, Z,  P_dB, cmap=cm.coolwarm, alpha=0.8, edgecolor='none')
    # 设置 60 仰角和 45 方位角
    # ax.view_init(elev=30, azim=30)
    ax.set_xlim(-np.pi/2, np.pi/2)
    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.scatter(phi_target, theta_target, np.max(np.max(P_dB)),color='blue', s=30, depthshade=True)
    # ax.plot_wireframe(Y, Z, P_dB, color='black', linewidth=0.1)
    ax.set_ylabel('Azimuth angle')
    ax.set_xlabel('Elevation angle')
    ax.set_zlabel('Gain')
    plt.tight_layout()
    ax.view_init(elev=30, azim=150)
    plt.show(dpi=1600)
    sio.savemat('res.mat',{'P_dB':P_dB})
    # print(theta_target, phi_target)
    # print(theta_target, phi_target)
    # Customize the z axis.

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # ax.scatter()
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

def show_nodes(index, Ny, name):

    plt.figure(figsize=(Ny, Ny))

    # 生成网格点
    x = np.arange(0, Ny, 1)
    y = np.arange(0, Ny, 1)
    X, Y = np.meshgrid(x, y)

    rows, cols = [], []
    for i in range(len(index)):
        col_temp, row_temp = one2two(index[i], Ny)
        rows.append(row_temp)
        cols.append(col_temp)
    # 绘制网格线
    plt.grid(True, linestyle='--', alpha=0.8,linewidth=1)

    # 在交点处绘制小圆点（但不是 scatter）
    # plt.plot(cols, rows, 'o', color='blue', markersize=6)
    plt.scatter(
        cols, rows,
        s=60,  # 点的大小
        facecolors='none',  # 无填充
        edgecolors='red',  # 边缘颜色
        linewidths=1.5,  # 边缘线宽
        marker='o'  # 圆形标记
    )
    plt.xlim(-0.5, Ny - 0.5)
    plt.ylim(-0.5, Ny - 0.5)
    plt.xlabel('y axis')
    plt.ylabel('z axis')
    # plt.title("6×6 Grid (Small Markers)")
    plt.savefig(name, dpi=1200)

def generate_isac_dataset(lamda, Wz, Wy, Nz, Ny,  K, J, Ns, ns, large_loss, Js):
    """
    generate isac dataset
    :param lamda: wavelength
    :param Wz:
    :param Wy:
    :param Nz:
    :param Ny:
    :param n_samples:
    :param K:
    :param Ns:
    :param ns:
    :param large_loss:
    :return:
    """
    dz = Wz / (Nz - 1)
    dy = Wy / (Ny - 1)
    # Hj = np.zeros((n_samples, J, Ns), dtype=complex)    # Hk @ E matrix
    Hk = np.zeros((K + J, Ns), dtype=complex)    # Hk @ E matrix
    b = list(range(Ns))
    combinations = get_combinations_recursive(b, ns)
    # for iter in range(n_samples):

    indexes = choice(combinations)
    E = selection_matrix(indexes, Ns)

    gk = 1 / np.sqrt(2) * (np.random.randn(Ns, K) + 1j * np.random.randn(Ns, K))
    Hk[:K,:] = (np.sqrt(large_loss) * Js.conj().T @ gk).T


    theta = (np.random.rand(J) - 0.5) * np.pi
    phi = (np.random.rand(J) - 0.5) * np.pi
    for jj in range(J):
        Hk[jj + K, :] = (np.sqrt(large_loss) *  two_dim_steering(dy, dz, theta[jj], phi[jj], Ny, Nz, lamda))[:,0]

    # Hk_all = np.hstack((np.real(Hk).reshape(n_samples,-1),  np.imag(Hk).reshape(n_samples,-1)))
    return Hk, theta, phi





