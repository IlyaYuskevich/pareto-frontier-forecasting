import numpy as np
from scipy.optimize import linprog


def main_loop(df, G, y, z, timescale):
    gy, gz = np.meshgrid(y, z)
    tau = timescale[0] * np.ones(G) - 50

    # create limits
    for j in range(G[0] - 1):
        for k in range(G[1]):
            # ly = (df.horsepower <= y[j + 1]) & (df.horsepower >= y[j])
            ly = df.horsepower >= y[j]
            lz = df.consumption <= z[k]
            # lz = df.consumption >= z[k]
            if df[ly & lz].size != 0:
                tau[j, k] = min(df[ly & lz].year)
            else:
                tau[j, k] = timescale[1]

            # lx = (df.acceleration <= x[i + 1]) & (df.acceleration >= x[i])
            # ly = df.horsepower >= y[j]
            # lz = df.consumption <= z[k]
            # if df[lx & ly & lz].size != 0:
            #     tau[i, j, k] = max([min(df[lx & ly & lz].year), tau[i, j, k]])

    tau = tau.ravel()
    # linear optimization
    c = np.ones(np.prod(G))
    A_ub = list(-np.eye(np.prod(G)))
    b_ub = list(-tau)
    A_eq = []

    # for i in np.arange(1, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(0, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i - 1, j, k] = -1
    #             A1[i, j, k] = 1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(0)

    #
    for j in np.arange(1, G[0]):
        for k in np.arange(0, G[1]):
            A1 = np.zeros(G)
            A1[j - 1, k] = 1
            A1[j, k] = -1
            A_ub.append(A1.ravel())
            b_ub.append(0)
    #
    for j in np.arange(0, G[0]):
        for k in np.arange(1, G[1]):
            A1 = np.zeros(G)
            A1[j, k - 1] = -1
            A1[j, k] = 1
            A_ub.append(A1.ravel())
            b_ub.append(0)
    # #
    # # # Global directions
    # for i in np.arange(1, G[0]):
    #     for j in np.arange(1, G[1]):
    #         for k in np.arange(1, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j - 1, k] = 1
    #             A1[i - 1, j, k - 1] = -1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(0)
    # #
    # Convexity in horsepower-cons
    for j in np.arange(1, G[0]):
        for k in np.arange(1, G[1]):
            A1 = np.zeros(G)
            A1[j, k] = -1
            A1[j - 1, k] = 1
            A1[j, k - 1] = 1
            A1[j - 1, k - 1] = -1
            A_ub.append(A1.ravel())
            b_ub.append(0)
    #
    #
    # # Convexity in cons-acc
    # for i in np.arange(1, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(1, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j, k] = 1
    #             A1[i - 1, j, k] = -1
    #             A1[i, j, k - 1] = -1
    #             A1[i - 1, j, k - 1] = 1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(0)

    # for i in np.arange(0, G[0]):
    #     for j in np.arange(1, G[1]):
    #         for k in np.arange(1, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j, k] = 1
    #             A1[i, j - 1, k - 1] = -1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(0)
    #
    #
    # for i in np.arange(1, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(1, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j, k] = 1
    #             A1[i - 1, j, k - 1] = -1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(0)

    # for i in np.arange(2, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(0, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i - 2, j, k] = -1
    #             A1[i - 1, j, k] = 2
    #             A1[i, j, k] = -1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(0)

    for j in np.arange(0, G[0]):
        for k in np.arange(2, G[1]):
            A1 = np.zeros(G)
            A1[j, k - 2] = -1
            A1[j, k - 1] = 2
            A1[j, k] = -1
            A_ub.append(A1.ravel())
            b_ub.append(0)

    for j in np.arange(2, G[0]):
        for k in np.arange(0, G[1]):
            A1 = np.zeros(G)
            A1[j - 2, k] = 1
            A1[j - 1, k] = -2
            A1[j, k] = 1
            A_ub.append(A1.ravel())
            b_ub.append(0)

    x = G[1]
    for j in np.arange(0, G[0]):
        for k in np.arange(x, G[1]):
            A1 = np.zeros(G)
            A1[j, k - 2] = 1.2
            A1[j, k - 1] = -2.2
            A1[j, k] = 1
            A_eq.append(A1.ravel())

    for j in np.arange(0, G[0]):
        for k in np.arange(2, x):
            A1 = np.zeros(G)
            A1[j, k - 2] = 1
            A1[j, k - 1] = -2.5
            A1[j, k] = 1.5
            A_eq.append(A1.ravel())

    # for i in np.arange(0, G[0]):
    #     for j in np.arange(0, G[1]):
    #         k = x-1
    #         A1 = np.zeros(G)
    #         A1[i, j, k - 2] = 1
    #         A1[i, j, k - 1] = -2
    #         A1[i, j, k] = 1
    #         A_ub.append(A1.ravel())
    #         b_ub.append(-1)
    #
    #         A1 = np.zeros(G)
    #         A1[i, j, k - 2] = 1
    #         A1[i, j, k - 1] = -2
    #         A1[i, j, k] = 1
    #         A_ub.append(A1.ravel())
    #         b_ub.append(1)

    for j in np.arange(0, G[0]):
        for k in np.arange(4, G[1]):
            A1 = np.zeros(G)
            A1[j, k - 1] = -1
            A1[j, k] = 1
            A_ub.append(A1.ravel())
            b_ub.append(-1)

    # for i in np.arange(0, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(2, 4):
    #             A1 = np.zeros(G)
    #             A1[i, j, k - 2] = -0.5  # 0.5
    #             A1[i, j, k - 1] = 1.5  # 1.5
    #             A1[i, j, k] = -1
    #             A_eq.append(A1.ravel())
    #
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.zeros(len(A_eq)),
                  method='interior-point',
                  # options={"lstsq": True, "presolve": False},
                  bounds=(timescale[0] - 20, timescale[1] + 100))
    print(res.message)

    result = res.x.reshape(G)
    return [gy, gz, result]
    # result = tau.reshape(G)
    # gz = np.exp(-np.exp(gz)) * physical_limit + 24
    # gz = np.exp(-np.exp(gz)) * physical_limit
    # gz = physical_limit - gz
    # df['consumption'] = np.exp(-np.exp(df['consumption'])) * physical_limit
    # df['consumption'] = physical_limit - df['consumption']
