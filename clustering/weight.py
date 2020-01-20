import numpy as np

class Weight:
    def __init__(self,
                 training_sample,
                 norm_training_sample=None,
                 ):
        self.__training_sample       = training_sample
        self.__norm_training_sample  = norm_training_sample
        self.__n                     = len(training_sample)

    def first_neighbor(self,
                       i,
                       point=None,
                       ):
        return self.k_neighbor(point=point, i=i, k=1)

    def k_neighbor(self,
                   i,
                   k,
                   point=None,
                   ):
        return self.k_neighbor_exp_weight(point=point, i=i, k=k, q=1)

    def k_neighbor_exp_weight(self,
                              i,
                              k,
                              q,
                              point=None,
                              ):
        if i >= k:
            return 0
        else:
            return q ** i
