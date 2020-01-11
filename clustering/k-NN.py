import data_2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    data1 = data_2.Data(600, sigma=1, mu=0).get()
    data2 = data_2.Data(600, sigma=1, mu=2).get()

    plt.title("Normal distribution", fontsize=20)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("y", fontsize=10)

    plt.scatter(data1['x'], data1['y'], s=5)
    plt.scatter(data2['x'], data2['y'], s=5)
    plt.show()
