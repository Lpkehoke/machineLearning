import matplotlib.pyplot as plt
import numpy as np

import data as Data
import metrics

class Knn:
    def __init__(self,
                 training_sample,
                 ):
        self.__training_sample   = training_sample
        self.__training_sample_n = []

    def normalize(self,
                  ):
        # max and min value
        max_x = np.amax(data[0]['x'] + data[1]['x'])
        max_y = np.amax(data[0]['y'] + data[1]['y'])
        min_x = np.amin(data[0]['x'] + data[1]['x'])
        min_y = np.amin(data[0]['y'] + data[1]['y'])

        # normalize
        self.__training_sample_n = []
        for i in (0, 1):
            self.__training_sample_n.append({
                'x': [(x - min_x) / (max_x - min_x) for x in data[i]['x']],
                'y': [(y - min_y) / (max_y - min_y) for y in data[i]['y']],
            })

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

        plt.scatter(data[0]['x'], data[0]['y'], s=5)
        plt.scatter(data[1]['x'], data[1]['y'], s=5)


if __name__ == '__main__':
    # get data
    data = []
    for mu in (0, 2):
        data.append(Data.Data(600, sigma=1, mu=mu).get())


    knn = Knn(data)
    knn.normalize()
    knn.drow_training_sample()
    knn.drow_training_sample_n()

    plt.show()
