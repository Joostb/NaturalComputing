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
        TPR.append(sum((df['values'] < value) & (df['label'] == 0)) / sum(df['label'] == 0))

        FPR.append(sum((df['values'] < value) & (df['label'] == 1)) / sum(df['label'] == 1))

    return [TPR, FPR]


def plot_ROC(Rs, names):
    for i, (TPR, FPR) in enumerate(Rs):
        plt.plot(FPR, TPR, label=names[i])
        plt.text(0.6, 0.5 - 0.075*(i+1), "AUC {}: {:.4f}".format(names[i], auc(FPR, TPR)))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.plot(np.linspace(0, max(FPR), len(FPR)), np.linspace(0, 1, len(TPR)), '--', label="Baseline")
    plt.legend()
    plt.show()


###########################
# Functions for exercises #
def do_it_all(paths, names):
    dfs = [get_dataframe(snd_cert_path, snd_label_path) for snd_cert_path, snd_label_path in paths]
    Rs = [compute_TPR_FPR(df) for df in dfs]
    plot_ROC(Rs, names)


if __name__ == "__main__":
    paths = [["results/ex_2/snd-cert_1_n7r4.out", "negative-selection/syscalls/snd-cert/snd-cert.1.labels"],
             ["results/ex_2/snd-cert_2_n7r4.out", "negative-selection/syscalls/snd-cert/snd-cert.2.labels"],
             ["results/ex_2/snd-cert_3_n7r4.out", "negative-selection/syscalls/snd-cert/snd-cert.3.labels"]]
    names = ["snd-cert.1", "snd_cert.2", "snd_cert.3"]
    do_it_all(paths, names)

    paths = [["results/ex_2/snd-unm_1_n7r4.out", "negative-selection/syscalls/snd-unm/snd-unm.1.labels"],
             ["results/ex_2/snd-unm_2_n7r4.out", "negative-selection/syscalls/snd-unm/snd-unm.2.labels"],
             ["results/ex_2/snd-unm_3_n7r4.out", "negative-selection/syscalls/snd-unm/snd-unm.3.labels"]]
    names = ["snd-unm.1", "snd_unm.2", "snd_unm.3"]
    do_it_all(paths, names)

