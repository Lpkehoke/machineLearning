import numpy as np
import json
import os


DEFAULT_SETTING = None

with open(os.path.dirname(os.path.realpath(__file__)) + '/setting.json', 'r') as setting:
    DEFAULT_SETTING = json.load(setting)['DEFAULT_SETTING']

if DEFAULT_SETTING is None:
    raise "wrong with setting.json"

NORMAL_DISTRIBUTION = DEFAULT_SETTING['NORMAL_DISTRIBUTION']
MU                  = NORMAL_DISTRIBUTION['mu']
SIGMA               = NORMAL_DISTRIBUTION['sigma']


class Data:
    def __init__(self,
                 num,
                 mu     = MU,
                 sigma  = SIGMA,
                 ):
        self.update(
            num     = num,
            mu      = mu,
            sigma   = sigma,
        )

    def __generate(self,
                   ):
        self.__x = np.random.normal(self.__mu, self.__sigma, self.__num)
        self.__y = np.random.normal(self.__mu, self.__sigma, self.__num)

    def update(self,
               num,
               mu       = MU,
               sigma    = SIGMA,
               ):
        self.__num    = num
        self.__mu     = mu
        self.__sigma  = sigma
        self.__x      = []
        self.__y      = []
        self.__generate()

    def get(self,
            ):
        return {
            'x': self.__x,
            'y': self.__y,
        }
