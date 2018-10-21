# -*- coding: utf-8 -*-
#=======================================
# A-Z ML
# Regression Practice
# Wine Quality Dataset
#=======================================


# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

# Preliminaries
#===================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import Practice Data
#===================

dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")


# Simple EDA
#===================

# plot all histograms in dataset

dataset.hist()
sns.pairplot(dataset)

dataset.describe()


# Data Preprocessing
#===================

# train test split based on outcome
train, test = train_test_split(dataset, random_state=1234, test_size=0.2, stratify=dataset["quality"])

# center and scale
scaler = StandardScaler()

X_train = scaler.fit_transform(train.iloc[:, :-1])
X_test = scaler.transform(test.iloc[:, :-1])

y_train = train.iloc[:,-1]
y_test = test.iloc[:,-1]













