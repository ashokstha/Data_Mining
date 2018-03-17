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

    mean1_x = np.mean(x[:100], dtype=np.float64)
    mean2_x = np.mean(x[100:], dtype=np.float64)
    mean1_y = np.mean(y[:100], dtype=np.float64)
    mean2_y = np.mean(y[100:], dtype=np.float64)

    mean1 = [mean1_x, mean1_y]
    mean2 = [mean2_x, mean2_y]

    print(mean1, mean2)

    std1_x = np.std(x[:100], dtype=np.float64)
    std2_x = np.std(x[100:], dtype=np.float64)
    std1_y = np.std(y[:100], dtype=np.float64)
    std2_y = np.std(y[100:], dtype=np.float64)

    std1 = [std1_x, std1_y]
    std2 = [std2_x, std2_y]

    print(std1, std2)
    print(np.mean(X_in[100:], axis=0))




    '''
    plt.scatter(x, y, c=label)
    plt.scatter(mean1_x,mean1_y, marker="*", c="#00FF00")
    plt.scatter(mean2_x, mean2_y, marker="*", c="#0000FF")
    plt.scatter(std1_x, std1_y + mean1_y, marker="^", c="#00FF00")
    plt.scatter(std2_x, std2_y + mean2_y, marker="^", c="#0000FF")

    plt.show()
    '''

    clf = LDA()
    clf.fit(X_in, label)

    print(clf.predict([[-0.8, -1]]))


if __name__ == "__main__":
    main()