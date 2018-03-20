#!/bin/python3

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def LDA(mean_1, mean_2, inv_cov):
    # co-efficients of decision variable
    b0 = (np.array(mean_1.dot(inv_cov)).dot(mean_1.T) - np.array(mean_2.dot(inv_cov)).dot(mean_2.T))
    b1, b2 = (inv_cov.dot(mean_2.T - mean_1.T) - (mean_1 - mean_2).dot(inv_cov))

    print("LDA Boundary Equation:\n{0} * X1 + {1} * X2 + {2} = 0\n".format(b1, b2, b0))

    x = np.linspace(-3, 1.1, 200)
    line = [-((b1 * x_val + b0) / b2) for x_val in x]
    plt.plot(x, line, color='red', linewidth=1)


def QDA(mean_1, mean_2, inv_cov1, inv_cov2):
    # co-efficients of decision variable
    b0 = (np.array(mean_1.dot(inv_cov1)).dot(mean_1.T) - np.array(mean_2.dot(inv_cov2)).dot(mean_2.T))
    bx1, bx2 = (inv_cov2.dot(mean_2.T) - inv_cov1.dot(mean_1.T) - mean_1.dot(inv_cov1) + mean_2.dot(inv_cov2))
    inv = inv_cov1 - inv_cov2
    bx1_sq = inv[0][0]
    bx2_sq = inv[1][1]
    bx1_x2 = inv[0][1] + inv[1][0]

    print(
        "QDA Boundary Equation:\n{0} * (X1 ^ 2) + {1} * (X2 ^ 2) + {2} * X1 * X2 + {3} * X1 + {4} * X2 + {5} = 0".format(
            bx1_sq,
            bx2_sq,
            bx1_x2,
            bx1,
            bx2,
            b0))

    x1_grid = np.linspace(-3, 1.1, 200)
    x2_grid = np.linspace(-1.5, 5, 200)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    func = bx1_sq * (X1 ** 2) + bx2_sq * (X2 ** 2) + bx1_x2 * X1 * X2 + bx1 * X1 + bx2 * X2 + b0
    plt.contour(X1, X2, func, [0])


def main():
    x = np.linspace(0, 1, 200)
    y = np.zeros_like(x, dtype=np.int32)
    x[0:100] = np.sin(4 * np.pi * x)[0:100]
    x[100:200] = np.cos(4 * np.pi * x)[100:200]
    y = 4 * np.linspace(0, 1, 200)  + 0.3 * np.random.randn(200)
    label = np.ones_like(x)
    label[0:100] = 0
    X_in = np.column_stack((x, y))

    mean_1 = np.mean(X_in[:100], axis=0)
    mean_2 = np.mean(X_in[100:], axis=0)

    cov_1 = np.cov(X_in[:100].T)
    cov_2 = np.cov(X_in[100:].T)
    cov = np.cov(X_in.T)

    inv_cov1 = np.linalg.inv(cov_1)
    inv_cov2 = np.linalg.inv(cov_2)
    inv_cov = np.linalg.inv(cov)

    LDA(mean_1, mean_2, inv_cov)
    QDA(mean_1, mean_2, inv_cov1, inv_cov2)

    plt.title("LDA & QDA Decision Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(x, y, c=label)
    plt.show()


if __name__ == "__main__":
    main()
