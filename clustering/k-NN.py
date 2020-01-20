import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

import data as Data
import metrics
import weight


class Knn:
    def __init__(self,
                 training_sample,
                 ):
        self.__training_sample      = training_sample
        self.__norm_training_sample = []
        self.__n                    = len(training_sample)

    def normalize_training_sample(self,
                                  ):
        # max and min value
        self.__max, self.__min = {}, {}
        for d in ('x', 'y'):
            self.__max[d] = []
            self.__min[d] = []
            for i in self.__training_sample:
                self.__max[d].append(np.amax(i[d]))
                self.__min[d].append(np.amin(i[d]))
            self.__max[d] = np.amax(self.__max[d])
            self.__min[d] = np.amin(self.__min[d])

        # normalize_training_sample
        self.__norm_training_sample = []
        for i in range(0, self.__n):
            self.__norm_training_sample.append({})
            for d in ('x', 'y'):
                self.__norm_training_sample[-1][d] = [(p - self.__min[d]) / (self.__max[d] - self.__min[d]) for p in self.__training_sample[i][d]]

    def normalize_point(self,
                        point,
                        ):
        if not self.__min or not self.__max:
            raise "first call normalize_training_sample"

        norm_point = {}
        for d in ('x', 'y'):
            norm_point[d] = (point[d] - self.__min[d]) / (self.__max[d] - self.__min[d])

        return norm_point

    def drow_training_sample(self,
                             ):
        self.__drow(self.__training_sample, figure = 0)

    def drow_training_sample_n(self,
                               ):
        self.__drow(self.__norm_training_sample, figure = 1)

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
    num_of_class  = 5
    num_of_points = 600

    data = []
    for i in range(0, num_of_class):
        data.append(Data.Data(num_of_points, sigma=1, mu=((i+1)*2)).get())

    knn = Knn(data)
    knn.normalize_training_sample()
    knn.drow_training_sample()
    knn.drow_training_sample_n()

    plt.show()
