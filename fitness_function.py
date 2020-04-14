import numpy as np
from numpy import pi
from numpy import cos

def booth_function(params):
    f = (params[0] + 2*params[1] - 7)**2 + (2*params[0] + params[1] - 5)**2
    return f

def rastrigin_function(params):
    A = 10
    n = len(params)
    f = A*n + np.sum(params**2 - A*cos(2*pi*params), axis=0)
    return f

def beale_function(params):
    f = (1.5 - params[0] + params[0]*params[1])**2 + (2.25 - params[0] + params[0]*(params[1]**2))**2 + (2.625 - params[0] + params[0]*(params[1]**3))**2
    return f

def happy_cat_function(params):
    pass

class FuncInf:
    NAME = ""
    MULTI_DIM_PARAMS = False
    NUM_PARAMS = 2
    GLOBAL_OPTIMA = (0, 0)
    DOMAIN = (0, 0)
    F_FUNC = None
    
    def __init__(self, name, num_params, multi_dim_params, domain, global_optima, func):
        self.NAME = name
        self.DOMAIN = domain
        self.NUM_PARAMS = num_params
        self.MULTI_DIM_PARAMS = multi_dim_params
        self.GLOBAL_OPTIMA = global_optima
        self.F_FUNC = func