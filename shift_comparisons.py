import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

for j in range(0, 1):

    # Box Plot
    data = []

    for i in range(6):
        fileResultsDetailsData = pd.read_csv("Tests 30x50x6000 V2\\GWO\\Both Shift\\GWO " + str(-i) +"\\experiment_details.csv")
        objective_name = "rastrigin"
        optimizer_name = "GWO"

        detailedData = fileResultsDetailsData[
            (fileResultsDetailsData["Optimizer"] == optimizer_name)
            & (fileResultsDetailsData["objfname"] == objective_name)
        ]
        detailedData = detailedData["Iter" + str(6000)]
        detailedData = np.array(detailedData).T.tolist()
        data.append(detailedData)

    # , notch=True
    box = plt.boxplot(data, patch_artist=True, labels=["0", "-1", "-2", "-3", "-4", "-5"])

    for patch in box["boxes"]:
        patch.set_facecolor('gray')
    for median in box['medians']:
        median.set_color('black')

    # plt.title("GWO_Modified Shrunk With Restarts Search Shift Comparison")
    # plt.ylabel("Fitness")
    # plt.legend(
    #     handles=box["boxes"],
    #     labels=["No Shift", "-1", "-2", "-3", "-4", "-5"],
    #     loc="upper right",
    #     bbox_to_anchor=(1.2, 1.02),
    # )
    plt.gca().set_ylim([0, 250])
    fig_name = "Tests 30x50x6000 V2\\GWO\\" + "/GWO Both Shift Comparison" + ".png"
    #plt.set_cmap(plt.get_cmap("gray"))
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()