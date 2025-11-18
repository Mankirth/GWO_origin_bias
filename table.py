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
    fileResultsDetailsData = pd.read_csv("CEC2022 30x50x2000 gray\\experiment_details.csv")
    objective_name = "F" + str(i+1)
    optimizer_name = "GWO"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(2000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)

    for t in range(30):
        detailedData[t] -= errors[i]
    gwoData = detailedData

    fileResultsDetailsData = pd.read_csv("CEC2022 30x50x2000 gray\\experiment_details.csv")
    objective_name = "F" + str(i+1)
    optimizer_name = "GWO_modified"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(2000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)

    for t in range(30):
        detailedData[t] -= errors[i]
    gwomData = detailedData.copy()

    total = 0
    sampleDiffs = 0
    diffs = np.zeros(30)
    for t in range(30):
        sampleDiffs += gwoData[t] - gwomData[t]
    sampleMean = sampleDiffs / 30
    #sampleMean = np.mean(gwomData) - np.mean(gwoData)
    for t in range(30):
        total += np.pow(((gwoData[t]-gwomData[t]) - sampleMean), 2)
    sampleStd = np.sqrt(total / (29))
    testStat = sampleMean/(sampleStd/np.sqrt(30))

    line.append([np.mean(gwoData).round(4), np.std(gwoData).round(4), np.mean(gwomData).round(4), np.std(gwomData).round(4), ((1 - (np.mean(gwomData)/np.mean(gwoData))) * 100).round(4), testStat.round(4)])
print(line)
cellText = line

# , notch=True

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(rowLabels=rows,
        colLabels=columns,
        cellText=cellText,
        loc='center')
fig.tight_layout()


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
# for patch in box["boxes"]:
#     patch.set_facecolor('gray')
# for median in box['medians']:
#     median.set_color('black')

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
fig_name = "CEC2022 30x50x2000 gray\\" + "table.png"
plt.savefig(fig_name, bbox_inches="tight")
plt.clf()
#plt.show()