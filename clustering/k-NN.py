import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

import data as Data
import metrics


class Knn:
    def __init__(self,
                 training_sample,
                 ):
        self.__training_sample   = training_sample
        self.__training_sample_n = []
        self.__n                 = len(training_sample)

    def normalize(self,
                  ):
        # max and min value
        max = {}
        min = {}
        for i in ('x', 'y'):
            max[i] = reduce((lambda x, y: np.amax(x[i] + y[i])), self.__training_sample)
            min[i] = reduce((lambda x, y: np.amin(x[i] + y[i])), self.__training_sample)

        # normalize
        self.__training_sample_n = []
        for i in range(0, self.__n):
            self.__training_sample_n.append({})
            for d in ('x', 'y'):
                self.__training_sample_n[-1][d] = [(x - min[d]) / (max[d] - min[d]) for x in self.__training_sample[i][d]]

    def drow_training_sample(self,
                             ):
        self.__drow(self.__training_sample, figure = 0)

    def drow_training_sample_n(self,
                             ):
        self.__drow(self.__training_sample_n, figure = 1)

    def __drow(self,
               data,
               figure = 0,
               ):
        plt.figure(figure)
        plt.title("Normal distribution", fontsize=20)
        plt.xlabel("x", fontsize=10)
        plt.ylabel("y", fontsize=10)

        for i in range(0, self.__n):
            plt.scatter(data[i]['x'], data[i]['y'], s=5)


if __name__ == '__main__':
    # get data
    num_of_class = 2
    data = []
    for i in range(0, 2):
        data.append(Data.Data(600, sigma=1, mu=((i+1)*2)).get())


    knn = Knn(data)
    knn.normalize()
    knn.drow_training_sample()
    knn.drow_training_sample_n()

    plt.show()
