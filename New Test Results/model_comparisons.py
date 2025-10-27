import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

for j in range(0, 6):

    # Box Plot
    data = []

    fileResultsDetailsData = pd.read_csv("Tests 30x50x6000\\GWO\\Origin Shift\\GWO " + str(-j) + "\\experiment_details.csv")
    objective_name = "rastrigin"
    optimizer_name = "GWO"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(4000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)

    fileResultsDetailsData = pd.read_csv("Tests 30x50x6000\\GWOM\\Origin Shift\\GWOM " + str(-j) + "\\experiment_details.csv")
    objective_name = "rastrigin"
    optimizer_name = "GWO_modified"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(4000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)

    fileResultsDetailsData = pd.read_csv("Tests 30x50x6000\\GWOM no init\\Origin Shift\\GWOMShift " + str(-j) + "\\experiment_details.csv")
    objective_name = "rastrigin"
    optimizer_name = "GWO_modified"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(4000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)

    fileResultsDetailsData = pd.read_csv("Tests 30x50x6000\\GWOMSR\\Origin Shift\\GWOMSR " + str(-j) + "\\experiment_details.csv")
    objective_name = "rastrigin"
    optimizer_name = "GWO_modified_shrunk"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(4000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)

    # , notch=True
    box = plt.boxplot(data, patch_artist=True, labels=["GWO", "GWOM", "GWOM_shift", "GWOMSR"])

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

    if(j == 0):
        plt.title("GWO Models Comparison (no shifts)")
    else:
        plt.title("GWO Models " + str(-j) + " Origin Shift Comparison")
    plt.ylabel("Fitness")
    plt.legend(
        handles=box["boxes"],
        labels=["GWO", "GWOM", "GWOM_shift", "GWOMSR"],
        loc="upper right",
        bbox_to_anchor=(1.2, 1.02),
    )
    plt.gca().set_ylim([0, 300])
    fig_name = "Tests 30x50x6000\\" + "/" + str(-j) + " origin_shift_function_comparison" + ".png"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()
    #plt.show()