import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

for j in range(0, 1):

    # Box Plot
    data = []

    for i in range(6):
        fileResultsDetailsData = pd.read_csv("Test Results\\GWOM Restart\\30x75x4000 " + str(-i) +"\\experiment_details.csv")
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
    box = plt.boxplot(data, patch_artist=True, labels=["No Shift", "-1", "-2", "-3", "-4", "-5"])

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

    plt.title("GWOM Shrunken W/ Search Space Shift & Restarts Comparison")
    plt.ylabel("Fitness")
    plt.legend(
        handles=box["boxes"],
        labels=["No Shift", "-1", "-2", "-3", "-4", "-5"],
        loc="upper right",
        bbox_to_anchor=(1.2, 1.02),
    )
    fig_name = "Test Results\\Rastrigin Function Shift Comparison Boxplots\\0-5 Comparisons\\" + "/boxplot-" + "GWOM_Restarts_function_shifts" + ".png"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()