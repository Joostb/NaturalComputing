import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

PREFIX_PATH = "results/"


def get_dataframe(english_path, tagalog_path):
    english = pd.read_csv(PREFIX_PATH + english_path, header=None)
    tagalog = pd.read_csv(PREFIX_PATH + tagalog_path, header=None)

    english_repeated = pd.DataFrame(['english' for _ in range(len(english))])
    english = pd.concat([english, english_repeated], axis=1, ignore_index=True)

    tagalog_repeated = pd.DataFrame(['tagalog' for _ in range(len(tagalog))])
    tagalog = pd.concat([tagalog, tagalog_repeated], axis=1, ignore_index=True)

    full_df = pd.concat([english, tagalog])
    full_df.columns = ['values', 'label']
    sorted_df = full_df.sort_values(by="values")

    return sorted_df


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
        TPR.append(sum((df['values'] > value) & (df['label'] == 'english')) / sum(df['label'] == 'english'))

        FPR.append(sum((df['values'] > value) & (df['label'] == 'tagalog')) / sum(df['label'] == 'tagalog'))

    return [TPR, FPR]


def plot_ROC(Rs):
    for i, (TPR, FPR) in enumerate(Rs):
        plt.plot(TPR, FPR, label=names[i])
        plt.text(0.6, 0.3 - 0.075*(i+1), "AUC {}: {:.4f}".format(names[i], auc(TPR, FPR)))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.plot(np.linspace(0, 1, len(TPR)), np.linspace(0, 1, len(TPR)), '--', label="Baseline")
    plt.legend()
    plt.show()


paths = [['english_r1.results', 'tagalog_r1.results'], ['english_r4.results', 'tagalog_r4.results'],
         ['english_r9.results', 'tagalog_r9.results']]
names = ["r = 1", "r = 4", "r = 9"]

dfs = [get_dataframe(english_path, tagalog_path) for english_path, tagalog_path in paths]
PRs = [compute_TPR_FPR(df) for df in dfs]
plot_ROC(PRs)
