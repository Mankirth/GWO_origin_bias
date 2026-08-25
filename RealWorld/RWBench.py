# An example code to test the wrapper for problems of the
#     Congress on Evolutionary Computation 2020
#     Competition on Real-World Constrained Optimization
# Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
#     Reshetnev Siberian State University of Science and Technology
#     Krasnoyarsk, Russian Federation
# Last change: 21/04/2025

from RealWorld import cec2020rwcw
import numpy as np

def RW1(xval):
    D = cec2020rwcw.global_D[1-1]
    ng = cec2020rwcw.global_gn[1-1]
    nh = cec2020rwcw.global_hn[1-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, 1, fval, gval, hval, D, ng, nh)

    return fval[0]

def RW45(xval):
    D = cec2020rwcw.global_D[45-1]
    ng = cec2020rwcw.global_gn[45-1]
    nh = cec2020rwcw.global_hn[45-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, 45, fval, gval, hval, D, ng, nh)

    return fval[0]

def RW46(xval):
    D = cec2020rwcw.global_D[46-1]
    ng = cec2020rwcw.global_gn[46-1]
    nh = cec2020rwcw.global_hn[46-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, 46, fval, gval, hval, D, ng, nh)

    return fval[0]

def RW47(xval):
    D = cec2020rwcw.global_D[47-1]
    ng = cec2020rwcw.global_gn[47-1]
    nh = cec2020rwcw.global_hn[47-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, 47, fval, gval, hval, D, ng, nh)

    return fval[0]

def RW48(xval):
    D = cec2020rwcw.global_D[48-1]
    ng = cec2020rwcw.global_gn[48-1]
    nh = cec2020rwcw.global_hn[48-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, 48, fval, gval, hval, D, ng, nh)

    return fval[0]

def RW49(xval):
    D = cec2020rwcw.global_D[49-1]
    ng = cec2020rwcw.global_gn[49-1]
    nh = cec2020rwcw.global_hn[49-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, 49, fval, gval, hval, D, ng, nh)

    return fval[0]

def RW50(xval):
    D = cec2020rwcw.global_D[50-1]
    ng = cec2020rwcw.global_gn[50-1]
    nh = cec2020rwcw.global_hn[50-1]

    fval = np.zeros(1)
    gval = np.zeros(ng)
    hval = np.zeros(nh)
    cec2020rwcw.cec2020rwc(xval, 50, fval, gval, hval, D, ng, nh)

    return fval[0]

def LowBounds(func_num):
    func_num = int(func_num[2:])
    D = cec2020rwcw.global_D[func_num-1]
    lowb = np.zeros(D)
    upb = np.zeros(D)

    cec2020rwcw.get_bounds(func_num, lowb, upb, D)
    return lowb

def UpBounds(func_num):
    func_num = int(func_num[2:])
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

def getFunctionDetails(function):
    return [function, LowBounds(function), UpBounds(function), cec2020rwcw.global_D[int(function[2:])-1]]



