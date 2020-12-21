import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

cmin = 1980
cmax = 2040


def plot3d(df, names, t, G):
    levels = np.arange(cmin, cmax, 5)
    fig = plt.figure(np.random.randint(100))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plt.grid()
    # print(pd.concat([names, df], axis=1, join_axes=[df.index]))
    ax1.scatter3D(df['horsepower'].values, df['acceleration'].values, df['consumption'].values, c=df['year'].values,
                  cmap='bone_r', vmin=cmin, vmax=cmax)
    # ax1.scatter3D(t[0].ravel(), t[1].ravel(), t[2].ravel(), c=t[3].ravel(), s=0.6, cmap='viridis',
    #               vmin=cmin, vmax=cmax)
    ax1.set_xlabel('horsepower')
    ax1.set_ylabel('acceleration')
    ax1.set_zlabel('consumption')

    ax2 = fig.add_subplot(2, 2, 2)
    plt.grid()
    ax2.scatter(df['consumption'][df['acceleration'] > 0].values,
                df['horsepower'][df['acceleration'] > 0].values,
                c=df['year'][df['acceleration'] > 0].values,
                cmap='bone_r', vmin=cmin, vmax=cmax)
    ax2.set_xlabel('consumption')
    ax2.set_ylabel('horsepower')
    ax2.set_xlim([0, 12])

    ax3 = fig.add_subplot(2, 2, 3)
    plt.grid()
    ax3.scatter(df['consumption'].values, df['acceleration'].values, c=df['year'].values,
                cmap='bone_r', vmin=cmin, vmax=cmax)
    ax3.set_xlabel('consumption')
    ax3.set_ylabel('acceleration')

    ax4 = fig.add_subplot(2, 2, 4)
    plt.grid()
    ax4.scatter(df['acceleration'].values, df['horsepower'].values, c=df['year'].values,
                cmap='bone_r', vmin=cmin, vmax=cmax)
    ax4.set_ylabel('horsepower')
    ax4.set_xlabel('acceleration')

    cs2 = ax2.contour(t[2][2, :, :], t[1][:, 2, :], t[3][7, :, :], cmap='bone_r', levels=levels)
    cs3 = ax3.contour(t[2][1, :, :].T, t[0][1, :, :].T, t[3][:, 0, :].T, cmap='bone_r', levels=levels)
    cs4 = ax4.contour(t[0][:, :, 6], t[1][:, :, 7], t[3][:, :, 4].T, cmap='bone_r',
                      levels=np.arange(cmin, cmax, 5))
    # ax2.contour(t[1][:, :], t[0][:, :], t[2][:, :].T, cmap='viridis', levels=np.arange(cmin, cmax, 5))
    # plt.colorbar(ax=ax2)
    ax2.clabel(cs2, levels[1::2], inline=1, fontsize=10, fmt='%.0f')
    ax3.clabel(cs3, levels[1::2], inline=1, fontsize=10, fmt='%.0f')
    ax4.clabel(cs4, levels[1::2], inline=1, fontsize=10, fmt='%.0f')
    plt.show()
