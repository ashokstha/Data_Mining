# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:40:12 2018

@author: Huaming

Homework 2

Submitted By: Ashok Kumar Shrestha
"""

import numpy as np
import collections as co
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

data = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], train_size=0.8, test_size=0.2, random_state = 5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.75, test_size=0.25, random_state = 5)

class KNN(object):
    def __init__(self):
        pass
    
    
    def train(self, X, y):
        """
        X = X_train
        y = y_train
        """
        self.X_train = X
        self.y_train = y

    def kNearestNeighbour(self, X_test, k = 1):
        #Calculate lables for each X_test
        distances = []
        lables = []
        for i in range(len(self.X_train)):
            distance = np.sqrt(np.sum(np.square(self.X_train[i,:] - X_test)))
            distances.append([i, distance])

        distances.sort(key = lambda x: x[1])
        #print distances

        for i in range(k):
            lables.append(self.y_train[distances[i][0]])

        predicts = co.Counter(lables).most_common()
        #print(predicts)

        if len(predicts) >= 2 and predicts[0][1] == predicts[1][1]:
            # No majority Vote
            return -1

        return predicts[0][0]


    def predict(self, X_test, k = 1): 
        """
        It takes X_test as input, and return an array of integers, which are the 
        class labels of the data corresponding to each row in X_test. 
        Hence, y_project is an array of lables voted by their corresponding 
        k nearest neighbors
        """
        if k > len(X_test):
            print("Error in K value")
            return

        y_project = []

        for i in range(len(X_test)):
            y_project.append(self.kNearestNeighbour(X_test[i,:],k))

        return y_project
            
    
    def report(self, X_test, y_test, k=1):
        """
        return the accurancy of the test data. 
        """
        predictions = self.predict(X_test, k)
        correct = 0

        for i in range(len(X_test)):
            if predictions[i] == y_test[i]:
                correct += 1

        accuracy = float(correct) / len(predictions) * 100
        return accuracy


    def k_validate(self, X_test, y_test):
        """
        plot the accuracy against k from 1 to a certain number so that one could pick the best k
        """
        k_val = []
        for i in range(1,len(y_test)//3):
            accuracy = self.report(X_test, y_test, i)
            k_val.append([i, accuracy])


        fig, ax = plt.subplots()
        plt.title("K Nearest Neighbour")
        x_axis = [x for x,y in k_val]
        y_axis = [y for x,y in k_val]
        ax.scatter(x_axis, y_axis, edgecolors=(0, 0, 0))
        ax.plot(x_axis, y_axis, '--', lw=1)
        ax.set_xlabel('K - Value')
        ax.set_ylabel('Accuracy')
        plt.show()

        k_val.sort(key=lambda x: x[1], reverse=True)
        return k_val[0][0]

knn_obj = KNN()
knn_obj.train(X_train, y_train)
k = knn_obj.k_validate(X_val, y_val)
print("K value: {0}".format(k))

result = knn_obj.report(X_test, y_test, k)
print("Accuracy: %0.2f %%" % result)
