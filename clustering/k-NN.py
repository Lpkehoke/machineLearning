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
        max, min = {}, {}
        for d in ('x', 'y'):
            max[d] = []
            min[d] = []
            for i in self.__training_sample:
                max[d].append(np.amax(i[d]))
                min[d].append(np.amin(i[d]))
            max[d] = np.amax(max[d])
            min[d] = np.amin(min[d])

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
    num_of_class = 5
    data = []
    for i in range(0, num_of_class):
        data.append(Data.Data(600, sigma=1, mu=((i+1)*2)).get())


    knn = Knn(data)
    knn.normalize()
    knn.drow_training_sample()
    knn.drow_training_sample_n()

    plt.show()
