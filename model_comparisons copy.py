import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

for j in range(0, 12):

    # Box Plot
    data = []
    errors = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]

    fileResultsDetailsData = pd.read_csv("CEC2022 30x50x2000 gray copy\\experiment_details.csv")
    objective_name = "F"+str(j+1)
    optimizer_name = "GWO"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(2000)]
    detailedData = np.array(detailedData).T.tolist()

    for t in range(30):
        detailedData[t] -= errors[j]

    data.append(detailedData)

    fileResultsDetailsData = pd.read_csv("CEC2022 30x50x2000 gray copy\\experiment_details.csv")
    objective_name = "F"+str(j+1)
    optimizer_name = "GWO_modified"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(2000)]
    detailedData = np.array(detailedData).T.tolist()

    for t in range(30):
        detailedData[t] -= errors[j]

    data.append(detailedData)

    # , notch=True
    box = plt.boxplot(data, patch_artist=True, labels=["GWO", "GWOM"])

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
    fig_name = "CEC2022 30x50x2000 gray copy\\" + "/boxplot-F" + str(j+1) + ".png"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()
    #plt.show()