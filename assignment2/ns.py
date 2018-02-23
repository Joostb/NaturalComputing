import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm


english = pd.read_csv('english.results', header=None)
tagalog = pd.read_csv('tagalog.results', header=None)

english_repeated = pd.DataFrame(['english' for _ in range(len(english))])
english = pd.concat([english,english_repeated], axis=1, ignore_index=True)

tagalog_repeated = pd.DataFrame(['tagalog' for _ in range(len(tagalog))])
tagalog = pd.concat([tagalog,tagalog_repeated], axis=1, ignore_index=True)

a = pd.concat([english,tagalog])

sorted_list = a.sort_values(by=0)
sorted_list.columns = ['values', 'label']

lower = []
higher = []

for  i, value_l in enumerate(tqdm (sorted_list.values)):

    value = value_l[0]
    lower.append(sum((sorted_list['values'] > value) & (sorted_list['label'] == 'english') ) / len(english))

    higher.append(sum((sorted_list['values'] > value) & (sorted_list['label'] == 'tagalog') ) / (len(tagalog) ))

plt.plot(  lower, higher)
plt.show()