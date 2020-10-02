from typing import List
import random
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

data = pd.read_csv('data/raw/properties_2017.csv')