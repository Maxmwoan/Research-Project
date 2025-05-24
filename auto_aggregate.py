import numpy as np
import scipy
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
warnings.filterwarnings('ignore')
import os

def agg_diff_states(trained_on, eval_on):
    mean_absolute_errors = []
    CI_sizes = []
    percent_inside_CI = []

    with open(f'experiment1/trained{trained_on}/eval{eval_on}/state1/data_summary.txt', 'r') as file1:
        # Read all lines into a list
        lines1 = file1.readlines()

    with open(f'experiment1/trained{trained_on}/eval{eval_on}/state8/data_summary.txt', 'r') as file2:
        # Read all lines into a list
        lines2 = file2.readlines()

    with open(f'experiment1/trained{trained_on}/eval{eval_on}/state42/data_summary.txt', 'r') as file3:
        # Read all lines into a list
        lines3 = file3.readlines()

    mean_absolute_errors.extend(eval(lines1[0]))
    mean_absolute_errors.extend(eval(lines2[0]))
    mean_absolute_errors.extend(eval(lines3[0]))

    percent_inside_CI.extend(eval(lines1[2]))
    percent_inside_CI.extend(eval(lines2[2]))
    percent_inside_CI.extend(eval(lines3[2]))

    CI_sizes.extend(eval(lines1[4]))
    CI_sizes.extend(eval(lines2[4]))
    CI_sizes.extend(eval(lines3[4]))

    print("Done")

    avg_MAE = np.average(mean_absolute_errors)
    stdev_MAE = np.std(mean_absolute_errors)
    avg_CI_sizes = np.average(CI_sizes)
    stdev_CI_size = np.std(CI_sizes)
    avg_percent_in_CI = np.average(percent_inside_CI)
    stdev_perc_in_CI = np.std(percent_inside_CI)

    print(avg_MAE)
    print(avg_CI_sizes)
    print(avg_percent_in_CI)
    print(stdev_MAE)

    def create_CI_plot():
        plt.hist(CI_sizes, bins=40, density=True)
        plt.title("Distribution of the Confidence Interval Sizes")

        kde = scipy.stats.gaussian_kde(CI_sizes)
        x_vals = np.linspace(min(CI_sizes), max(CI_sizes), 500)
        plt.plot(x_vals, kde(x_vals), 'r-', linewidth=2, label='KDE')

    def create_MEA_PercentCI():
        plt.hist2d(x=mean_absolute_errors,
                   y=percent_inside_CI,
                   bins=(40, 40),
                   cmap="inferno")
        plt.colorbar(label='Count')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Percentage Inside Confidence Interval')
        plt.title('2D Histogram: MAE vs CI accuracy')

    def create_MAE_hist():
        plt.hist(mean_absolute_errors, bins=40, density=True)
        plt.title("Distribution of the Mean absolute error")

        kde = scipy.stats.gaussian_kde(mean_absolute_errors)
        x_vals = np.linspace(min(mean_absolute_errors), max(mean_absolute_errors), 500)
        plt.plot(x_vals, kde(x_vals), 'r-', linewidth=2, label='KDE')

    def create_percent_in_CI():
        plt.hist(percent_inside_CI, bins=40, density=True)
        plt.title("Percent inside CI")

        kde = scipy.stats.gaussian_kde(percent_inside_CI)
        x_vals = np.linspace(min(percent_inside_CI), max(percent_inside_CI), 500)
        plt.plot(x_vals, kde(x_vals), 'r-', linewidth=2, label='KDE')

    # create_CI_plot()
    # plt.show()
    #
    # create_MAE_hist()
    # plt.show()
    #
    # create_percent_in_CI()
    # plt.show()
    #
    # create_MEA_PercentCI()
    # plt.show()

    output_dir = f"experiment1/trained{trained_on}/eval{eval_on}/aggregate"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/data_summary_trained{trained_on}_eval{eval_on}.txt", "w") as f:
        f.write(mean_absolute_errors.__str__())
        f.write("\n")
        f.write("Mean absolute errors. avg:" + str(avg_MAE) + ", stdev:" + str(stdev_MAE))
        f.write("\n")
        f.write(percent_inside_CI.__str__())
        f.write("\n")
        f.write("Percentage of curve inside of confidence interval. avg:" + str(avg_percent_in_CI) + ", stdev:" + str(
            stdev_perc_in_CI))
        f.write("\n")
        f.write(CI_sizes.__str__())
        f.write("\n")
        f.write("Confidence interval sizes. avg:" + str(avg_CI_sizes) + ", stdev:" + str(stdev_CI_size))

    create_CI_plot()
    plt.savefig(f"{output_dir}/ci_sizes.png")
    plt.close()

    create_MAE_hist()
    plt.savefig(f"{output_dir}/MAE_hist")
    plt.close()

    create_percent_in_CI()
    plt.savefig(f"{output_dir}/PercentInCI_hist")
    plt.close()

    create_MEA_PercentCI()
    plt.savefig(f"{output_dir}/MAE_vs_PercentInCI")
    plt.close()

if __name__ == "__main__":
    groups = {
        "SVC": [0, 1, 2, 3],
        "Trees": [4, 5, 20, 21, 22],
        "NB": [14, 15, 16, 17],
        "Neighbors": [18, 19],
        "DA": [12, 13],
        "Linear": [6, 7, 8, 9, 10],
        "nn": [11],
        "Dummy": [23]
    }

    for trained_on in groups.keys():
        for eval_on in groups.keys():
            print(f"{trained_on}, {eval_on}")
            agg_diff_states(trained_on, eval_on)

