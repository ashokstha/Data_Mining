#!/bin/python3

import sys
import numpy as np

threshold = 1e-5

def pageRanking(G):
    # Page ranking function returns page-ranking vector
    pageRanks = np.array([1, 0, 0, 0, 0])
    oldRanks = np.array([0, 0, 0, 0, 0])

    while np.sum(np.abs(pageRanks - oldRanks)) > threshold:
        oldRanks = pageRanks
        pageRanks = np.dot(G,pageRanks)

    return pageRanks


if __name__ == "__main__":
    #randomly generate 5x5 matrix
    x = np.random.random((5, 5))
    #divide each element of matrix with sum of col values to make sum of col val 1
    x = x/x.sum(axis=0)
    print(x)

    pageRank = pageRanking(x)
    print([rank for rank in pageRank])