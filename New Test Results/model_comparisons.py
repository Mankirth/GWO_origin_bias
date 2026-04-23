import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

# Box Plot
data = []

fileResultsDetailsData = pd.read_csv("aa PSO docd" + "\\experiment_details.csv")
objective_name = "rastrigin"
optimizer_name = "PSO"
detailedData = fileResultsDetailsData[
    (fileResultsDetailsData["Optimizer"] == optimizer_name)
    & (fileResultsDetailsData["objfname"] == objective_name)
]
detailedData = detailedData["Iter" + str(5999)]
detailedData = np.array(detailedData).T.tolist()
data.append(detailedData)

fileResultsDetailsData = pd.read_csv("aa EPSO original" + "\\experiment_details.csv")
objective_name = "rastrigin"
optimizer_name = "EPSO"
detailedData = fileResultsDetailsData[
    (fileResultsDetailsData["Optimizer"] == optimizer_name)
    & (fileResultsDetailsData["objfname"] == objective_name)
]
detailedData = detailedData["Iter" + str(5999)]
detailedData = np.array(detailedData).T.tolist()
data.append(detailedData)

fileResultsDetailsData = pd.read_csv("aa FPSO original" + "\\experiment_details.csv")
objective_name = "rastrigin"
optimizer_name = "FPSO"
detailedData = fileResultsDetailsData[
    (fileResultsDetailsData["Optimizer"] == optimizer_name)
    & (fileResultsDetailsData["objfname"] == objective_name)
]
detailedData = detailedData["Iter" + str(5999)]
detailedData = np.array(detailedData).T.tolist()
data.append(detailedData)

# , notch=True
box = plt.boxplot(data, patch_artist=True, labels=["PSO", "EPSO", "FPSO"])

colors = [
    "#5c9eb7",
    "#f77199",
    "#cf81d2",
    "#4a5e6a",
    "#f45b18",
    "#ffbd35",
    "#6ba5a1",
    "#fcd1a1",
    "#c3ffc1",
    "#68549d",
    "#1c8c44",
    "#a44c40",
    "#404636",
]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

fig_name = "aa PSO Comparisons\\" + "/" + " origin_shift_function_comparison" + ".png"
plt.savefig(fig_name, bbox_inches="tight")
plt.clf()
#plt.show()