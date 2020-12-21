from visualizastion import plot3d
from optimization import main_loop
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('data/dataset.xlsx', names=['price', 'year', 'horsepower', 'acceleration', 'consumption',
                                               'CO2', 'mpg'])
names = pd.read_excel('data/names.xlsx', names=['model'])
# df = df[df['year'] > 1985]
df = df[df['consumption'] < 20]
df = df[~np.isnan(np.array(df['acceleration']))]

## create grid
G = [10, 10, 10]
x = np.linspace(0, 1 * df['acceleration'].max(), G[0])
y = np.linspace(40, df['horsepower'].max(), G[1])
z = np.linspace(0, 1 * df['consumption'].max(), G[2])
backwardTest = False  # backward testing flag
opt = False  # opt flag

df2 = df[df['year'] < 1996]

if opt:
    res1, tau1 = main_loop(df, G, x, y, z, [min(df['year']), max(df['year'])])

    if backwardTest:
        res2, tau2 = main_loop(df2, G, x, y, z, [min(df['year']), max(df['year'])])
# res1 = main_loop(df, G, y, z, [min(df['year']), max(df['year'])])
# res2 = main_loop(df2, G, y, z, [min(df['year']), max(df['year'])])
#

    plot3d(df, names, res1, G)
    if backwardTest:
        plot3d(df2, names, res2, G)

    plt.figure(2)
    plt.plot(res1[3][6, 1, :], res1[2][6, 1, :], '-o')
    if backwardTest:
        plt.plot(res2[3][6, 1, :], res2[2][6, 1, :], '-o')
        plt.grid()

    if backwardTest:
        plt.figure(3)
        plt.scatter(res2[3].ravel(), (res2[3] - res1[3]).ravel())
        plt.xlabel('Year')
        plt.ylabel('Estimated error (years)')
        plt.ylim([-20, 20])
        plt.grid()
    #
    # plt.figure(2)
    # plt.scatter(res2[2].ravel(), (res2[2] - res1[2]).ravel())


    # plot3d(df, names, [res1[0:3], tau1] G)
    # plot3d(df2, names, [res2[0:3], tau2], G)

horsepower_interval = df['horsepower'].max() - df['horsepower'].min()
acceleration_interval = df['acceleration'].max() - df['acceleration'].min()
consumption_interval = df['consumption'].max() - df['consumption'].min()

print('Horsepower domain', horsepower_interval)
print('Acceleration domain', acceleration_interval)
print('Consumption domain', consumption_interval)

n_points = []
std_points = []
# df = df[df.index % 1000 == 0]

for i in range(G[0] - 1):
    for j in range(G[1] - 1):
        for k in range(G[2] - 1):
            lx = df.acceleration.between(x[i], x[i + 1])
            ly = df.horsepower.between(y[j], y[j + 1])
            lz = df.consumption.between(z[k], z[k + 1])
            size = df[lx & ly & lz].size
            std = df[lx & ly & lz].year.std()
            if (size > 3) & ~np.isnan(std):
                n_points.append(size)
                std_points.append(std)

std_mean = np.array(std_points)/np.sqrt(np.array(n_points))
