import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

plt.ioff()
# Box Plot
data = []

for j in range(0, 12):

    fileResultsDetailsData = pd.read_csv("GWOM with shift\\experiment_details.csv")
    objective_name = "F"+str(j+1)
    optimizer_name = "GWO_modified"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Shift"]
        
    detailedData = np.array(detailedData).T.tolist()

    for i in range(30):
        detailedData[i] = detailedData[i][1:-1]
        detailedData[i] = str.split(detailedData[i], " ")
        while '' in detailedData[i]:
            detailedData[i].remove('')
        for u in range(10):
            if "\\n" in detailedData[i][u]:
                detailedData[i][u][0:-2]
            detailedData[i][u] = float(detailedData[i][u])
        detailedData[i] = abs(np.mean(detailedData[i]))

    data.append(detailedData)

# , notch=True
box = plt.boxplot(data, patch_artist=True, labels=["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"])

# colors = [
#     "#5c9eb7",
#     "#f77199",
#     "#cf81d2",
#     "#4a5e6a",
#     "#f45b18",
#     "#ffbd35",
#     "#6ba5a1",
#     "#fcd1a1",
#     "#c3ffc1",
#     "#68549d",
#     "#1c8c44",
#     "#a44c40",
#     "#404636",
# ]
# for patch, color in zip(box["boxes"], colors):
#     patch.set_facecolor(color)
for patch in box["boxes"]:
    patch.set_facecolor('gray')
for median in box['medians']:
    median.set_color('black')

# if(j == 0):
#     plt.title("GWO Models Comparison (no shifts)")
# else:
#     plt.title("GWO Models " + str(-j) + " Origin Shift Comparison")
# plt.ylabel("Fitness")
# plt.legend(
#     handles=box["boxes"],
#     labels=["GWO", "GWOM", "GWOM_shift", "GWOMSR"],
#     loc="upper right",
#     bbox_to_anchor=(1.2, 1.02),
# )
#plt.gca().set_ylim([0, 300])
fig_name = "GWOM with shift\\" + "/f comparison.png"
plt.savefig(fig_name, bbox_inches="tight")
plt.clf()
#plt.show()