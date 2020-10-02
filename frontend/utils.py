from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

def plot_roc(y_val, y_pred):
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_val, y_pred)

    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(false_positive_rate, true_positive_rate, color='b')
    plt.title('ROC Curve')

def plot_pr(y_val, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
    plt.figure(figsize=(15, 10))
    plt.grid()

    plt.plot(thresholds, precision[1:], color='r', label='Precision')
    plt.plot(thresholds, recall[1:], color='b', label='Recall')

    plt.gca().invert_xaxis()
    plt.legend()

def evaluate(y_val, y_pred):
    plot_roc(y_val, y_pred)
    plot_pr(y_val, y_pred)

    return roc_auc_score(y_val, y_pred)
