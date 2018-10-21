# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np


#======================================
# Melbourne Pedestrian Data PLotting
#======================================

dat = pd.read_csv("Parking_bay_arrivals_and_departures_2014.csv")

# Exploratory data analysis
#===========================


# create features
#------------------

# time from ArrivalTime

dat['ArrivalTime'] = pd.to_datetime(dat['ArrivalTime'])

# day of week from ArrivalTime

dat['ArrivalHour'] = dat['ArrivalTime'].dt.hour
dat['ArrivalDayOfWeek'] = dat['ArrivalTime'].dt.dayofweek
dat['ArrivalDayMonth'] = dat['ArrivalTime'].dt.month


dat = dat[(dat["ArrivalDayMonth"] == 9) | (dat["ArrivalDayMonth"] == 10)]
# Simple Logistic Regression
#===========================

dat['ArrivalHour'] = dat['ArrivalHour'].astype('category')
dat['ArrivalDayOfWeek'] = dat['ArrivalDayOfWeek'].astype('category')

dat.columns

cats = ['StreetName','ArrivalHour','ArrivalDayOfWeek', 'BetweenStreet1 Description', 'BetweenStreet2 Description', 'SideName']
dat_model_dummies = pd.get_dummies(dat[cats])


c = pd.DataFrame(dat_model_dummies.columns)

# train and test split.

X_train = dat_model_dummies[dat["ArrivalDayMonth"] == 9] 
X_test = dat_model_dummies[dat["ArrivalDayMonth"] == 10] 

y_train = dat['InViolation'][dat["ArrivalDayMonth"] == 9] 
y_test = dat['InViolation'][dat["ArrivalDayMonth"] == 10] 

# logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 


#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
#clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)

clf = LogisticRegression(penalty='l2', C=0.1, class_weight="balanced")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score  

print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))  


# Predict using time of day/ day of week /  month etc
#========================================

pred_set = dat[dat['ArrivalDayMonth'] == 10]
pred_set['pred'] = y_pred

print(confusion_matrix(pred_set['InViolation'], pred_set['pred']))

# 7am Monday morning:

seven_mon = pred_set[(pred_set['ArrivalDayOfWeek'] == 0) & (pred_set['ArrivalHour'] == 7)]
seven_mon['pred'] = seven_mon['pred'].astype(int)
seven_mon['InViolation'] = seven_mon['InViolation'].astype(int)


violations = seven_mon.groupby('StreetName')['pred'].agg(['sum','count'])
violations_rows = seven_mon.groupby('StreetName')['pred'].count()

actuals = seven_mon.groupby('StreetName')['InViolation'].agg(['sum','count'])



roc_auc_score(y_test, y_pred)



    