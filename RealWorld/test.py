# An example code to test the wrapper for problems of the
#     Congress on Evolutionary Computation 2020
#     Competition on Real-World Constrained Optimization
# Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
#     Reshetnev Siberian State University of Science and Technology
#     Krasnoyarsk, Russian Federation
# Last change: 21/04/2025

from cec2020rwcw import global_D, global_gn, global_hn, get_bounds, cec2020rwc
import numpy as np

func_num = 45
D = global_D[func_num-1]
ng = global_gn[func_num-1]
nh = global_hn[func_num-1]
lowb = np.zeros(D)
upb = np.zeros(D)

get_bounds(func_num, lowb, upb, D)
print(int((10000 * D) / 50))

xval = [2000,100,4000,36.52718493,0,20,8.36616895]
fval = np.zeros(1)
gval = np.zeros(ng)
hval = np.zeros(nh)
cec2020rwc(xval, func_num, fval, gval, hval, D, ng, nh)

print("HERE: ",xval,fval)
