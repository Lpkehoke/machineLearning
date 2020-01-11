import numpy as np

class Metrics:
    def Euclidean(a, b):
        return np.sqrt( np.square(a['x'] - b['x']) + np.square(a['y'] - b['y']) )
