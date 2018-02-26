import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

PREFIX_PATH = "results/"


def get_dataframe(file_path, label_path):
    abnormality_list = []
    with open(file_path, 'r') as f:
        for line in f:
            _line = line.split()
            mean = np.mean([float(number) for number in _line])
            abnormality_list.append(mean)

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(int(line))

    data = np.vstack((abnormality_list, labels))
    data = data.transpose()

    df = pd.DataFrame(data, columns=["values", "label"])
    df = df.sort_values(by="values")

    return df


def compute_TPR_FPR(df):
    """
    The idea is that we plot the TPR against the FPR. Where FP is that we wrongly classify a text as english.
    In the loop we calculate the TPR and FPR as follows:
        1. Pick the value of the sorted_df as the threshold.
        2. Everything above this threshold is deemed English, below tagalog
        3. True Positive: Correctly classified as English
        4. False Positive: Wrongly classified as English
    :param df:
    :return:
    """

    TPR = []
    FPR = []

    for value_l in tqdm(df.values):
        value = value_l[0]
        TPR.append(sum((df['values'] > value) & (df['label'] == 0)) / sum(df['label'] == 0))

        FPR.append(sum((df['values'] > value) & (df['label'] == 1)) / sum(df['label'] == 1))

    return [TPR, FPR]


def plot_ROC(Rs, names):
    for i, (TPR, FPR) in enumerate(Rs):
        plt.plot(TPR, FPR, label=names[i])
        plt.text(0.6, 0.62 - 0.075*(i+1), "AUC {}: {:.4f}".format(names[i], auc(TPR, FPR)))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.plot(np.linspace(0, 1, len(TPR)), np.linspace(0, 1, len(TPR)), '--', label="Baseline")
    plt.legend()
    plt.show()


###########################
# Functions for exercises #
def main():
    df = get_dataframe("results/ex_2/snd-cert_1_n7r7.out", "negative-selection/syscalls/snd-cert/snd-cert.1.labels")
    Rs = [compute_TPR_FPR(df)]
    plot_ROC(Rs, ["snd-cert.1"])


if __name__ == "__main__":
    main()
