# Python wrapper of the C++ code of the
#     Congress on Evolutionary Computation 2020
#     Competition on Real-World Constrained Optimization
# Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
#     Reshetnev Siberian State University of Science and Technology
#     Krasnoyarsk, Russian Federation
# Last change: 21/04/2025

from ctypes import CDLL, POINTER, c_int, c_double
import numpy as np
import nt

global_D = np.array([9,11,7,6,9,38,48,2,3,3,7,7,5,10,7,14,3,4,4,2,5,9,5,7,4,22,10,10,4,3,4,5,30,118,153,158,126,126,126,76,74,86,86,30,25,25,25,30,30,30,59,59,59,59,64,64,64])
global_gn = np.array([0,0,14,1,2,0,0,2,1,3,4,9,3,10,11,15,4,4,5,3,8,10,8,7,7,86,3,9,1,8,0,6,30,0,0,0,0,0,0,0,0,0,0,91,24,24,24,29,29,29,14,14,14,14,0,0,0])
global_hn = np.array([8,9,0,4,4,32,38,0,1,0,4,0,0,0,0,0,0,0,0,0,0,1,3,0,0,0,0,0,0,0,0,0,0,108,148,148,116,116,116,76,74,76,76,0,1,1,1,1,1,1,1,1,1,1,6,6,6])

def get_bounds(func_num, lowb, upb, nx, dll_path=CDLL('RealWorld\cec2020rwc.so',winmode=0x8)):
    functions = dll_path
    func_num_type = c_int
    lowb_pointer_type = POINTER(c_double * nx)
    upb_pointer_type  = POINTER(c_double * nx)
    functions.get_bounds.argtypes = [func_num_type, lowb_pointer_type, upb_pointer_type]
    functions.get_bounds.restype = None
    lowb_ctype = (c_double * nx)()
    upb_ctype  = (c_double * nx)()
    for i in range(nx):
        lowb_ctype[i] = 0    
        upb_ctype[i] = 0
    functions.get_bounds(func_num, lowb_pointer_type(lowb_ctype), upb_pointer_type(upb_ctype))
    for i in range(nx):
        lowb[i] = lowb_ctype[i]
        upb[i]  = upb_ctype[i]

def cec2020rwc(xval, func_num, fval, gval, hval, nx, ng, nh, dll_path=CDLL('RealWorld\cec2020rwc.so', winmode=0x8)):
    functions = dll_path
    xval_pointer_type = POINTER(c_double * nx)    
    func_num_type = c_int
    fval_pointer_type = POINTER(c_double)
    gval_pointer_type = POINTER(c_double * ng)
    hval_pointer_type = POINTER(c_double * nh)   
    
    functions.cec20_func.argtypes = [xval_pointer_type, func_num_type, fval_pointer_type, gval_pointer_type, hval_pointer_type] 
    functions.cec20_func.restype = None
    x_ctype = (c_double * nx)()
    for i, value in enumerate(xval):
        x_ctype[i] = value
    f_ctype = (c_double * 1)()
    for i in range(1):
        f_ctype[i] = 0
    g_ctype = (c_double * ng)()
    for i in range(ng):
        g_ctype[i] = 0
    h_ctype = (c_double * nh)()
    for i in range(nh):
        h_ctype[i] = 0
    
    functions.cec20_func(xval_pointer_type(x_ctype), func_num, fval_pointer_type(f_ctype), gval_pointer_type(g_ctype), hval_pointer_type(h_ctype))
    fval[0] = f_ctype[0]
    for i in range(ng):
        gval[i] = g_ctype[i]
    for i in range(nh):
        hval[i] = h_ctype[i]
    return fval
