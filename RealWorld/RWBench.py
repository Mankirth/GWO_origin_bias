# An example code to test the wrapper for problems of the
#     Congress on Evolutionary Computation 2020
#     Competition on Real-World Constrained Optimization
# Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
#     Reshetnev Siberian State University of Science and Technology
#     Krasnoyarsk, Russian Federation
# Last change: 21/04/2025

from RealWorld import cec2020rwcw
import numpy as np

def Eval(xval, func_num):
    D = cec2020rwcw.global_D[func_num-1]
    ng = cec2020rwcw.global_gn[func_num-1]
    nh = cec2020rwcw.global_hn[func_num-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, func_num, fval, gval, hval, D, ng, nh)

    return fval

def LowBounds(func_num):
    D = cec2020rwcw.global_D[func_num-1]
    lowb = np.zeros(D)
    upb = np.zeros(D)

    cec2020rwcw.get_bounds(func_num, lowb, upb, D)
    return lowb

def UpBounds(func_num):
    D = cec2020rwcw.global_D[func_num-1]
    lowb = np.zeros(D)
    upb = np.zeros(D)

    cec2020rwcw.get_bounds(func_num, lowb, upb, D)
    return upb

def Dim(func_num):
    return cec2020rwcw.global_D[func_num-1]

def GetRandomStart(func_num):
    D = cec2020rwcw.global_D[func_num-1]
    lowb = np.zeros(D)
    upb = np.zeros(D)

    cec2020rwcw.get_bounds(func_num, lowb, upb, D)

    return np.random.uniform(lowb,upb)


