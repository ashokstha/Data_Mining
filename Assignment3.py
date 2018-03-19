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
    print("means: \nmean1: {0}\nmean2: {1}\n".format(mean_1, mean_2))

    cov_1 = np.cov(X_in[:100].T)
    cov_2 = np.cov(X_in[100:].T)
    cov = np.cov(X_in.T)
    print("Covariance:\ncov1:\n {0}\ncov2:\n {1}\ncov:\n {2}\n".format(cov_1, cov_2, cov))

    inv_cov1 = np.linalg.inv(cov_1)
    inv_cov2 = np.linalg.inv(cov_2)
    inv_cov = np.linalg.inv(cov)
    print("Inverse Covariance:\ninv_cov1: \n{0}\ninv_cov2:\n {1}\ninv_cov:\n {2}\n".format(inv_cov1, inv_cov2, inv_cov))

    '''
        LDA calculations: using matrices
    '''
    [m11, m12] = mean_1
    [m21, m22] = mean_2

    [[a, b], [c, d]] = inv_cov

    # co-efficients of decision variable
    b0 = (np.array(mean_1.dot(inv_cov)).dot(mean_1.T) - np.array(mean_2.dot(inv_cov)).dot(mean_2.T))
    b1, b2 = (inv_cov.dot(mean_2.T - mean_1.T) - (mean_1 - mean_2).dot(inv_cov))

    print("LDA: coeffiecients: ", b0, b1, b2)
    print("Equation:\n {0} * x1 + {1} * x2 + {2} = 0\n".format(b1, b2, b0))

    # x_val = 0
    # y_val = (b1 * x_val + b0)/b2
    line = [-((b1 * x_val + b0) / b2) for x_val in x]

    plt.scatter(x, y, c=label)
    plt.plot(x, line, color='red', linewidth=1)
    # plt.show()

    '''
        QDA Calculations: using matrices
    '''
    b0 = (np.array(mean_1.dot(inv_cov1)).dot(mean_1.T) - np.array(mean_2.dot(inv_cov2)).dot(mean_2.T))
    bx1, bx2 = (inv_cov2.dot(mean_2.T) - inv_cov1.dot(mean_1.T) - mean_1.dot(inv_cov1) + mean_2.dot(inv_cov2))
    inv = inv_cov1 - inv_cov2
    bx1_sq = inv[0][0]
    bx2_sq = inv[1][1]
    bx1_x2 = inv[0][1] + inv[1][0]

    #b0 = 0
    print("QDA Equation:\n {0} * x1^2 + {1} * x2^2 + {2} * x1 * x2 + {3} * x1 + {4} * x2 + {5} = 0\n".format(bx1_sq,
                                                                                                             bx2_sq,
                                                                                                             bx1_x2,
                                                                                                             bx1,
                                                                                                             bx2, b0))
    for i in x:
        '''
        qda_eqn = "{0} + {1} * i**2 + {2} * i + {3} + {4} * i + {5}".format(bx1_sq * (i * i), bx2_sq,
                                                                            bx1_x2 * i, bx1 * i,
                                                                            bx2, b0)
                                                                            '''
        qda_eqn = "{0} + {1} * i**2 + {2} * i".format(bx1_sq * (i * i) + b0 + bx1 * i, bx2_sq,
                                                                    bx2+ bx1_x2 * i)
        y = eval(qda_eqn)
        # print(qda_eqn,y)
        plt.scatter(i, y, color='green', marker=".")

    plt.show()


if __name__ == "__main__":
    main()
