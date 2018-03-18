#!/bin/python3

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn

sklearn.__version__
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def main():
    x = np.linspace(0, 1, 200)
    y = np.zeros_like(x, dtype=np.int32)
    x[0:100] = np.sin(4 * np.pi * x)[0:100]
    x[100:200] = np.cos(4 * np.pi * x)[100:200]
    y = 4 * np.linspace(0, 1, 200) + 0.3 * np.random.randn(200)
    label = np.ones_like(x)
    label[0:100] = 0
    X_in = np.column_stack((x, y))

    mean_1 = np.mean(X_in[:100], axis=0)
    mean_2 = np.mean(X_in[100:], axis=0)
    print("means: ", mean_1, mean_2)

    '''
    std_1 = np.std(X_in[:100], axis=0)
    std_2 = np.std(X_in[100:], axis=0)
    print("std: ", std_1, std_2)
    '''

    cov_1 = np.cov(X_in[:100].T)
    cov_2 = np.cov(X_in[100:].T)
    cov = np.cov(X_in.T)
    print("Covariance:", cov_1, cov_2, cov)

    inv_cov1 = np.linalg.inv(cov_1)
    inv_cov2 = np.linalg.inv(cov_2)
    inv_cov = np.linalg.inv(cov)
    print("Inverse:", inv_cov1, inv_cov2, inv_cov)

    #mean_1 = [0, 0]
    #mean_2 = [3, 4]
    #inv_cov = [[2, -1], [-1, 3]]

    [m11, m12] = mean_1
    [m21, m22] = mean_2

    [[a, b], [c, d]] = inv_cov

    # co-efficients of decision variable
    b0 = ((m11 ** 2 - m21 ** 2) * a + (m12 ** 2 - m22 ** 2) * d + (m11 * m12 - m21 * m22) * (b + c))/2
    b1 = -(2 * (m11 - m21) * a + (m12 - m22) * (b + c))/2
    b2 = -(2 * (m12 - m22) * d + (m11 - m21) * (b + c))/2

    print("coeffiecients: ", b0, b1, b2)

    # plt.scatter(x, y, c=label)
    '''
    plt.scatter(mean1_x,mean1_y, marker="*", c="#00FF00")
    plt.scatter(mean2_x, mean2_y, marker="*", c="#0000FF")
    plt.scatter(std1_x, std1_y + mean1_y, marker="^", c="#00FF00")
    plt.scatter(std2_x, std2_y + mean2_y, marker="^", c="#0000FF")
    '''

    # plt.show()


if __name__ == "__main__":
    main()
