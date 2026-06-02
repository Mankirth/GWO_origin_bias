import numpy as np
import math
def cec20_func(x,prob_k):
    f = None
    g = None
    h = None
    # cec20_func Constrained Optimization Test Suite 
    # Abhishek Kumar (email: abhishek.kumar.eee13@iitbhu.ac.in, Indian Institute of Technology (BHU), Varanasi) 

    # x -----> ps X D where 'ps': number of population and 'D': Dimension of
    # the problem.
    # f -----> Objective Function Value.
    # g -----> Inequality Consstraints Value ps X ng where 'ng': number of
    # inequality constraints.
    # h -----> Equality Constraints Value ps X nh where 'nh': number of
    # equality constraints.
    # prob_k -> Index of problem.

## Industrial Chemical Processes		
def benchmark1(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## Heat Exchanger Network Design (case 1)
    f = np.power(np.multiply(35, x[:,1]), 0.6) + np.power(np.multiply(35,x[:,2]),0.6)
    g = np.zeros(ps,1)
    h[:,1] = np.multiply(np.multiply(200, x[:,1]), x[:,4]) - x[:,3]
    h[:,2] = np.multiply(np.multiply(200, x[:,2]), x[:,6]) - x[:,5]
    h[:,3] = x[:,3] - 10000*(x[:,7]-100)
    h[:,4] = x[:,5] - 10000*(300-x[:,7])
    h[:,5] = x[:,3] - 10000*(600-x[:,8])
    h[:,6] = x[:,5] - 10000*(900-x[:,9])
    h[:,7] = np.multiply(x[:,4], math.log(abs(x[:,8]-100)+1e-8)) - np.multiply(x[:,4], math.log((600-x[:,7])+1e-8)) - x[:,8]+x[:,7]+500
    h[:,8] = np.multiply(x[:,6], math.log(abs(x[:,9]-x[:,7])+1e-8)) - np.multiply(x[:,6], math.log(600))-x[:,9]+x[:,7]+600
    return f

def benchmark2(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## Heat Exchanger Network Design (case 2)
    f = np.power((np.divide(x[:,1], (120*x[:,4]))),0.6) + np.power(np.divide(x[:,2] , (80*x[:,5])),0.6) + np.power(np.divide(x[:,3], (40*x[:,6])),0.6)
    g = np.zeros(ps,1)
    h[:,1] = x[:,1]-np.multiply(1e4, (x[:,7]-100))
    h[:,2] = x[:,2]-np.multiply(1e4, (x[:,8]-x[:,7]))
    h[:,3] = x[:,3]-np.multiply(1e4, (500-x[:,8]))
    h[:,4] = x[:,1]-np.multiply(1e4, (300-x[:,9]))
    h[:,5] = x[:,2]-np.multiply(1e4, (400-x[:,1]))
    h[:,6] = x[:,3]-np.multiply(1e4, (600-x[:,1]))
    h[:,7] = np.multiply(x[:,4], math.log(math.abs(x[:,9]-100)+1e-8) - np.multiply(x[:,4], math.log(300-x[:,7])) + 1e-8) - x[:,9]-x[:,7]+400
    h[:,8] = np.multiply(x[:,5], math.log(math.abs(x[:,10]-x[:,7])+1e-8))- np.multiply(x[:,5], math.log(abs(400-x[:,8])+1e-8)) - x[:,10]+x[:,7]-x[:,8]+400
    h[:,9] = np.multiply(x[:,6], math.log(math.abs(x[:,11]-x[:,8])+1e-8))- np.multiply(x[:,6], math.log(100))-x[:,11]+x[:,8]+100
    return f

def benchmark3(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## Optimal Operation of Alkylation Unit
    f = np.multiply(-1.715, x[:,1]) - np.multiply(np.multiply(0.035, x[:,1]),x[:,6]) - np.mulyiply(4.0565, x[:,3]) - np.multiply(10.0, x[:,2]) + np.multiply(np.multiply(0.063, x[:,3]), x[:,5])
    h = np.zeros(ps,1)
    g[:,1] = np.multiply(np.power(np.multiply(0.0059553571, x[:,6]),2), x[:,1]) + np.multiply(0.88392857, x[:,3]) - np.multiply(np.multiply(0.1175625, x[:,6]), x[:,1])-x[:,1]
    g[:,2] = np.multiply(np.multiply(np.multiply(1.1088, x[:,1])+0.1303533,x[:,1]),x[:,6]) - np.power(np.multiply(np.multiply(0.0066033, x[:,1]),x[:,6]),2) - x[:,3]
    g[:,3] = np.power(np.multiply(6.66173269, x[:,6]),2) + np.multiply(np.multiply(172.39878, x[:,5]-56.596669),x[:,4]) - np.multiply(191.20592, x[:,6]) - 10000
    g[:,4] = np.multiply(1.08702, x[:,6]) + np.multiply(0.32175, x[:,4]) - np.power(np.multiply(0.03762, x[:,6]),2) - x[:,5] + 56.85075
    g[:,5] = np.multiply(np.multiply(np.multiply(0.006198, x[:,7]),x[:,4]),x[:,3])+2462.3121*x[:,2]-25.125634*x[:,2]*x[:,4]-x[:,3]*x[:,4]
    g[:,6] = np.multiply(161.18996, x[:,3])*x[:,4]+5000.0*x[:,2]*x[:,4]-489510.0*x[:,2]-x[:,3]*x[:,4]*x[:,7]
    g[:,7] = np.multiply(0.33, x[:,7])-x[:,5]+44.333333
    g[:,8] = np.multiply(0.022556, x[:,5])-0.007595*x[:,7]-1.0
    g[:,9] = np.multiply(0.00061, x[:,3])-0.0005*x[:,1]-1.0
    g[:,10]= np.multiply(0.819672, x[:,1])-x[:,3]+0.819672
    g[:,11]= np.multiply(24500.0, x[:,2])-250.0*x[:,2]*x[:,4]-x[:,3]*x[:,4]
    g[:,12]= np.multiply(1020.4082, x[:,4])*x[:,2]+1.2244898*x[:,3]*x[:,4]-100000*x[:,2]
    g[:,13]= np.multiply(6.25, x[:,1])*x[:,6]+6.25*x[:,1]-7.625*x[:,3]-100000
    g[:,14]= np.multiply(1.22, x[:,3])-x[:,6]*x[:,1]-x[:,1]+1.0
    return f

def benchmark4(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## Reactor Network Design (RND)
    k1 = 0.09755988
    k2 = np.multiply(0.99,k1)
    k3 = 0.0391908
    k4 = np.multiply(0.9,k3)
    f = -x[:,4]
    h[:,1] = x[:,1]+k1*x[:,2]*x[:,5]-1
    h[:,2] = x[:,2]-x[:,1]+k2*x[:,2]*x[:,6]
    h[:,3] = x[:,3]+x[:,1]+k3*x[:,3]*x[:,5]-1
    h[:,4] = x[:,4]-x[:,3]+x[:,2]-x[:,1]+k4*x[:,4]*x[:,6]
    g[:,1] = x[:,5]^0.5+x[:,6]^0.5-4
    return f

def benchmark5(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## Haverly's Pooling Problem
    f1 = np.multiply(9, x[:, 1]) + np.multiply(15, x[:,2])
    f2 = np.multiply(6, x[:,3])
    f3 = np.multiply(16, x[:,4])
    f4 = np.multiply(10, x[:, 5] + x[:,6])
    f = -(f1-f2-f3-f4)
    g[:,1] = np.multiply(x[:,9], x[:,7]) + np.multiply(2, x[:,5]) - np.multiply(2.5, x[:,1])
    g[:,2] = np.multiply(x[:,9], x[:,8]) + np.multiply(2, x[:,6]) - np.multiply(1.5, x[:,2])
    h[:,1] = x[:,7]+x[:,8]-x[:,3]-x[:,4]
    h[:,2] = x[:,1]-x[:,7]-x[:,5]
    h[:,3] = x[:,2]-x[:,8]-x[:,6]
    h[:,4] = np.multiply(x[:,9], x[:,7]) + np.multiply(x[:,9], x[:,8]) - np.multiply(3, x[:,3]) -x[:,4]
    return f


## Power Electronic Problems		

def benchmark45(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## SOPWM for 3-level Invereters
    m = 0.32
    s = (-np.ones(1,25))^np.arange(2,26)
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97]
    for i in range(ps):
        i += 1
        su = 0
        for j in range(31):
            j += 1
            su2 = 0
            for l in range(D):
                l += 1
                su2 = su2 + s(l)*math.cos(k(j)*x(i,l)*math.pi/180)
            
            su = su + su2^2/k(j)^4
        
        f(i,1) = (su)^0.5/(sum(1/k^4))^0.5
    
    g = np.zeros(ps,D-1)
    for i in range(D-1):
        i += 1
        g[:,i] = x[:,i]-x[:,i+1]+1e-6
    
    h = sum(s*math.cos(x*math.pi/180),2)-m
    return f



def benchmark46(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## SOPWM for 5-level Inverters
    m = 0.32
    s = [1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1]
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97]
    for i in range(ps):
        i += 1
        su = 0
        for j in range(31):
            j += 1
            su2 = 0
            for l in range(D):
                l += 1
                su2 = su2 + s(l)*math.cos(k(j)*x(i,l)*math.pi/180)
            
            su = su + su2^2/k(j)^4
        
        f(i,1) = 0.5*(su)^0.5/(sum(1/k^4))^0.5
    
    g = np.zeros(ps,D-1)
    for i in range(D-1):
        i += 1
        g[:,i] = x[:,i]-x[:,i+1]+1e-6
    
    h = sum(s*math.cos(x*math.pi/180),2)-2*m
    return f


def benchmark47(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## SOPWM for 7-level Inverters
    m = 0.36
    s = [1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1,1,1]
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97]
    for i in range(ps):
        i += 1
        su = 0
        for j in range(31):
            j += 1
            su2 = 0
            for l in range(D):
                l += 1
                su2 = su2 + s(l)*math.cos(k(j)*x(i,l)*math.pi/180)
            
            su = su + su2^2/k(j)^4
        
        f(i,1) = 1/3*(su)^0.5/(sum(1/k^4))^0.5
    
    g = np.zeros(ps,D-1)
    for i in range(D-1):
        i += 1
        g[:,i] = x[:,i]-x[:,i+1]+1e-6
    
    h = sum(np.multiply(s, math.cos(x*math.pi/180)),2)-3*m    
    return f


def benchmark48(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## SOPWM for 9-level Inverters
    m = 0.32
    s = [1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,1]
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97]
    for i in range(ps):
        i += 1
        su = 0
        for j in range(31):
            j += 1
            su2 = 0
            for l in range(D):
                l += 1
                su2 = su2 + s(l)*math.cos(k(j)*x(i,l)*math.pi/180)
            
            su = su + su2^2/k(j)^4
        
        f(i,1) = 1/4*(su)^0.5/(sum(1/k^4))^0.5
    
    g = np.zeros(ps,D-1)
    for i in range(D-1):
        i += 1
        g[:,i] = x[:,i]-x[:,i+1]+1e-6
    
    h = sum(s*math.cos(x*math.pi/180),2)-4*m 
    return f


def benchmark49(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## SOPWM for 11-level Inverters
    m = 0.3333
    s = [1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,-1,-1,1,-1,-1]
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97]
    for i in range(ps):
        i += 1
        su = 0
        for j in range(31):
            j += 1
            su2 = 0
            for l in range(D):
                l += 1
                su2 = su2 + s(l)*math.cos(k(j)*x[i,l]*math.pi/180)
            
            su = su + su2^2/k(j)^4
        
        f(i,1) = 1/5*(su)^0.5/(sum(1/k^4))^0.5
    
    g = np.zeros(ps,D-1)
    for i in range(D-1):
        g[:,i] = x[:,i]-x[:,i+1]+1e-6
    
    h = sum(s*math.cos(x*math.pi/180),2)-5*m 
    return f


def benchmark50(x):
    ps = x.shape[0]
    D = x.shape[1]
    h = np.zeros(ps, 30)
    ## SOPWM for 13-level Inverters
    m = 0.32
    s = [1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1]
    k = [5,7,11,13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73,77,79,83,85,91,95,97]
    for i in range(ps):
        i += 1
        su = 0
        for j in range(31):
            j += 1
            su2 = 0
            for l in range(D):
                su2 = su2 + s(l)*math.cos(k(j)*x(i,l)*math.pi/180)
        
            su = su + su2^2/k(j)^4
    
        f(i,1) = 1/6*(su)^0.5/(sum(1/k^4))^0.5

    g = np.zeros(ps,D-1)
    for i in range(D-1):
        i += 1
        g[:,i] = x[:,i]-x[:,i+1]+1e-6

    h = sum(s*math.cos(x*math.pi/180),2)-6*m 

    g= np.transpose(g)
    h= np.transpose(h)
    return f


def cec22_test_func(x, nx, mx, func_num):
  global OShift, M, y, z, x_bound, ini_flag, n_flag, func_flag, SS
  
  OShift = None 
  M = None 
  y = None
  z = None 
  x_bound = None 
  ini_flag = 0
  n_flag = None 
  func_flag = None
  SS = None
  cf_num = 10
  if (func_num < 1)|(func_num > 12):
    print('\nError: Test function %d is not defined.\n' %func_num)
  if ini_flag == 1:
    if (n_flag != nx)|(func_flag != func_num):
      ini_flag = 0

  if ini_flag == 0:
    del(M)
    del(OShift)
    del(y)
    del(z)
    del(x_bound)
    y = [0]*nx
    z = [None]*nx
    x_bound = [100.0]*nx

    if (nx!=2|nx!=10|nx!=20):
      print("\nError: Test functions are only defined for D=2,10,20.\n")

    if (nx==2)&(func_num==6 | func_num==7 | func_num==8):
      print("\nError:  NOT defined for D=2.\n")
      
    
    # Load M matrix
    
    FileName = 'input_data/M_%d_D%d.txt'%(func_num, nx)
    try:
      M = np.loadtxt(FileName)
    except:
      print("\n Error: Cannot open M_%d_D%d.txt for reading \n" %(func_num,nx))
    del(FileName)
    
    # Shift data
    FileName = "input_data/shift_data_%d.txt" %func_num
    try:
      OShift_temp = np.loadtxt(FileName)
    except:
      print("\n Error: Cannot open shift_data_%d.txt for reading \n" %func_num)
#    if OShift == None:
#      print("\nError: there is insufficient memory available!\n")
    del(FileName)
    if (func_num < 9):
        OShift = np.zeros((nx,))
        for i in range(nx):
            OShift[i] = OShift_temp[i]
    else:
        
        OShift = np.zeros((cf_num-1,nx))
        for i in range(cf_num-1):
            for j in range(nx):
                OShift[i,j] = OShift_temp[i,j]
        OShift = np.reshape(OShift, (cf_num-1)*nx)

    
    
    if (func_num >= 6) & (func_num <=8):
        FileName = "input_data/shuffle_data_%d_D%d.txt" %(func_num, nx)
        try:
          SS = np.loadtxt(FileName)
        except:
          print("\n Error: Cannot open shuffle_data_%d_D%d.txt for reading \n" %(func_num, nx))
  
        del(FileName)

    n_flag = nx
    func_flag = func_num
    ini_flag = 1
    f = np.zeros((mx,))
    for i in range(mx):
      if func_num == 1:
        ff = benchmark1(x, nx, OShift, M, 1, 1)
        f[i] = ff + 300.0
        break
      elif func_num == 2:
        ff = benchmark2(x,  nx, OShift, M, 1, 1)
        f[i] = ff + 400.0
        break
      elif func_num == 3:
        ff = benchmark3(x, nx, OShift, M, 1, 1)
        f[i] = ff + 600.0
        break
      elif func_num == 4:
        ff = benchmark4(x,  nx, OShift, M, 1, 1)
        f[i] = ff + 800.0
        break
      elif func_num == 5:
        ff = benchmark5(x,  nx, OShift, M, 1, 1)
        f[i] = ff + 800.0
        break
      elif func_num == 45:
        ff = benchmark45(x, nx, OShift, M, 1, 1)
        f[i] = ff + 900.0
        break
      elif func_num == 46:
        ff = benchmark46(x,  nx, OShift, M, SS, 1, 1)
        f[i] = ff + 1800.0
        break
      elif func_num == 47:
        ff = benchmark47(x, nx, OShift, M, SS, 1, 1)
        f[i] = ff + 2000.0
        break
      elif func_num == 48:
        ff = benchmark48(x, nx, OShift, M, SS, 1, 1)
        f[i] = ff + 2200.0
        break
      elif func_num == 49:
        ff = benchmark49(x,  nx, OShift, M, 1, 1)
        f[i] = ff + 2300.0
        break
      elif func_num == 50:
        ff = benchmark50(x,  nx, OShift, M, 1, 1)
        f[i] = ff + 2400.0
        break
      else:
        print("\nError: There are only 10 test functions in this test suite!\n")
        f[i] = 0.0
        break
    
    return f
    
class cec2022_func():
  
    def __init__(self, func_num):

        self.func = func_num
        

    def values(self, x):
        
        (nx,) = x.shape
        mx = 1

        ObjFunc = np.zeros(mx)
        for i in range(mx):
            ObjFunc[i] = cec22_test_func(x, nx, 1, self.func)
        
        self.ObjFunc = ObjFunc[0] if mx == 1 else ObjFunc
        
        return self
    
def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 10],
        "F2": ["F2", -100, 100, 10],
        "F3": ["F3", -100, 100, 10],
        "F4": ["F4", -100, 100, 10],
        "F5": ["F5", -100, 100, 10],
        "F6": ["F6", -100, 100, 10],
        "F7": ["F7", -100, 100, 10],
        "F8": ["F8", -100, 100, 10],
        "F9": ["F9", -100, 100, 10],
        "F10": ["F10", -100, 100, 10],
    }
    return param.get(a, "nothing")

def F1(x):
    return cec2022_func(1).values(x).ObjFunc

def F2(x):
    return cec2022_func(2).values(x).ObjFunc

def F3(x):
    return cec2022_func(3).values(x).ObjFunc

def F4(x):
    return cec2022_func(4).values(x).ObjFunc

def F5(x):
    return cec2022_func(5).values(x).ObjFunc

def F6(x):
    return cec2022_func(6).values(x).ObjFunc

def F7(x):
    return cec2022_func(7).values(x).ObjFunc

def F8(x):
    return cec2022_func(8).values(x).ObjFunc

def F9(x):
    return cec2022_func(9).values(x).ObjFunc

def F10(x):
    return cec2022_func(10).values(x).ObjFunc