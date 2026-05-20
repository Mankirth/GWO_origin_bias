import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.ioff()

# Box Plot
data = []
line = []

columns = ('GWO Mean', 'GWO StdDev', 'GWOM Mean', 'GWOM StdDev', "%-Diff", "Paired T-Test")
rows = ['1','2','3','4','5','6','7','8','9','10','11','12']
errors = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]
for i in range(12):
    detailedData = []
    for y in range(30):
        with open('CEC2022 - FTT-PSO\CEC2022 - FTT-PSO\CEC22-F'+str(i+1)+'\Mcfstpso'+str(y)+'-CEC22-F'+str(i+1)+'-20.txt') as f:
            mini = f.readline()
            for x in f:
                if float(x) < float(mini):
                    mini = x
            detailedData.append(float(mini) - errors[i])
    detailedData = np.array(detailedData).T.tolist()

    print("F" + str(i+1) + ": mean: " + str(np.mean(detailedData).round(4)) + " std: " + str(np.std(detailedData).round(4)))