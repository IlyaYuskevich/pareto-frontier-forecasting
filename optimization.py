import numpy as np
from scipy.optimize import linprog


def main_loop(df, G, x, y, z, timescale, flag = False):
    gx, gy, gz = np.meshgrid(x, y, z)
    tau = timescale[0] * np.ones(G) - 50
    c = np.ones(G)


    # create limits
    for i in range(G[0]):
        for j in range(G[1]):
            for k in range(G[2]):
                lx = df.acceleration <= (x[i] + 0.4*(x[2]-x[1]))
                # ly = (df.horsepower <= y[j + 1]) & (df.horsepower >= y[j])
                ly = df.horsepower >= (y[j] - 0.4*(y[2]-y[1]))
                lz = df.consumption <= (z[k] + 0.4*(z[2]-z[1]))
                if df[lx & ly & lz].size != 0:
                    tau[i, j, k] = min(df[lx & lz].year)
                else:
                    tau[i, j, k] = timescale[0]+40
                    c[i, j, k] = 0

    for i in range(G[0]):
        for j in range(G[1]):
            for k in np.arange(G[1] - 1, 0, -1):
                if (tau[i, j, k] - tau[i, j, k - 1]) > 0:
                    tau[i, j, k - 1] = tau[i, j, k]
    #
    for j in range(G[1]):
        for k in range(G[2]):
            for i in np.arange(G[0] - 1, 0, -1):
                if (tau[i, j, k] - tau[i - 1, j, k]) > 0:
                    tau[i-1, j, k] = tau[i, j, k]

    # for i in range(G[0]):
    #     for k in range(G[2]):
    #         for j in range(G[1]):
    #             if (tau[i, j, k] - tau[i, j - 1, k]) < 0:
    #                 tau[i, j, k] = tau[i, j - 1, k]

    tau_o = tau
    print(tau)
    tau = tau.ravel()
    c = c.ravel()
    # linear optimization
    A_ub = list(-np.eye(np.prod(G)))
    b_ub = list(-tau)
    A_eq = []

    for i in np.arange(1, G[0]):
        for j in np.arange(0, G[1]):
            for k in np.arange(0, G[2]):
                A1 = np.zeros(G)
                A1[i - 1, j, k] = -1
                A1[i, j, k] = 1
                A_ub.append(A1.ravel())
                b_ub.append(0)

    #
    for i in np.arange(0, G[0]):
        for j in np.arange(1, G[1]):
            for k in np.arange(0, G[2]):
                A1 = np.zeros(G)
                A1[i, j - 1, k] = 1
                A1[i, j, k] = -1
                A_ub.append(A1.ravel())
                b_ub.append(0)
    #
    for i in np.arange(0, G[0]):
        for j in np.arange(0, G[1]):
            for k in np.arange(1, G[2]):
                A1 = np.zeros(G)
                A1[i, j, k - 1] = -1
                A1[i, j, k] = 1
                A_ub.append(A1.ravel())
                b_ub.append(0)
    # # #
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
    # for i in np.arange(0, G[0]):
    #     for j in np.arange(1, G[1]):
    #         for k in np.arange(1, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j, k] = 1
    #             A1[i, j - 1, k] = -1
    #             A1[i, j, k - 1] = -1
    #             A1[i, j - 1, k - 1] = 1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(-0.1)
    q=1.2
    for i in np.arange(0, G[0]):
        for j in np.arange(1, G[1]):
            for k in np.arange(1, G[2]):
                A1 = np.zeros(G)
                A1[i, j, k] = 2*q
                A1[i, j - 1, k] = -(1+q)
                A1[i, j, k - 1] = -(1+q)
                A1[i, j - 1, k - 1] = 2
                A_ub.append(A1.ravel())
                b_ub.append(-0)
    #
    #
    # Convexity in cons-acc
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

    # for i in np.arange(0, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(1, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j, k - 1] = 1
    #             A1[i, j, k] = -1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(0)

    # for i in np.arange(0, G[0]):
    #     for j in np.arange(2, G[1]):
    #         for k in np.arange(0, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j - 2, k] = 1
    #             A1[i, j - 1, k] = -2
    #             A1[i, j, k] = 1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(-0.5)

    x = G[2] - 4
    for i in np.arange(0, G[0]):
        for j in np.arange(0, G[1]):
            for k in np.arange(x, G[2]):
                A1 = np.zeros(G)
                A1[i, j, k - 2] = 1.1
                A1[i, j, k - 1] = -2.1
                A1[i, j, k] = 1
                A_eq.append(A1.ravel())

    for i in np.arange(0, G[0]):
        for j in np.arange(0, G[1]):
            for k in np.arange(2, x):
                A1 = np.zeros(G)
                A1[i, j, k - 2] = 1
                A1[i, j, k - 1] = -2.7
                A1[i, j, k] = 1.7
                A_eq.append(A1.ravel())

    # x = 4
    # for i in np.arange(x, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(0, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i - 2, j, k] = 1.1
    #             A1[i - 1, j, k] = -2.1
    #             A1[i, j, k] = 1
    #             A_eq.append(A1.ravel())
    #
    # for i in np.arange(2, x):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(0, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i - 2, j, k] = 1
    #             A1[i - 1, j, k] = -2.2
    #             A1[i, j, k] = 1.2
    #             A_eq.append(A1.ravel())

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

    # for i in np.arange(0, G[0]):
    #     for j in np.arange(0, G[1]):
    #         for k in np.arange(4, G[2]):
    #             A1 = np.zeros(G)
    #             A1[i, j, k - 1] = -1
    #             A1[i, j, k] = 1
    #             A_ub.append(A1.ravel())
    #             b_ub.append(-0.5)
    if flag:
        for i in np.arange(0, G[0]):
            for j in np.arange(0, G[1]):
                for k in np.arange(4, G[2]):
                    A1 = np.zeros(G)
                    A1[i, j, k - 1] = 1
                    A1[i, j, k] = -1
                    A_ub.append(A1.ravel())
                    b_ub.append(5)

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
    return [gx, gy, gz, result], tau_o
    # result = tau.reshape(G)
    # gz = np.exp(-np.exp(gz)) * physical_limit + 24
    # gz = np.exp(-np.exp(gz)) * physical_limit
    # gz = physical_limit - gz
    # df['consumption'] = np.exp(-np.exp(df['consumption'])) * physical_limit
    # df['consumption'] = physical_limit - df['consumption']
