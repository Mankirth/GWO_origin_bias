import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.ioff()

# Box Plot
data = []

fileResultsDetailsData = pd.read_csv("PSOS Rastrigin Tracked\\experiment_details.csv")
objective_name = "rastrigin"
optimizer_name = "FPSO"
detailedData = fileResultsDetailsData[
    (fileResultsDetailsData["Optimizer"] == optimizer_name)
    & (fileResultsDetailsData["objfname"] == objective_name)
]
detailedData = detailedData["Iter" + str(6000)]
detailedData = np.array(detailedData).T.tolist()
data.append(detailedData)

gwoData = detailedData
print(np.mean(gwoData).round(4))
print(np.std(gwoData).round(4))