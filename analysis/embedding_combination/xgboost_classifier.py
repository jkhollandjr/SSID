import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.metrics import confusion_matrix

device = torch.device("cuda")
train_dataset = np.load('similarity_train.npy')
train_labels = np.load('labels_train.npy')

test_dataset = np.load('similarity_test.npy')
test_labels = np.load('labels_test.npy')

model = xgb.XGBRegressor(objective="binary:logistic")
model.fit(train_dataset, train_labels)

y_pred_prob = model.predict(test_dataset)
thresholds = [.000000001, .001, .01, .05, .1, .25, .5, .75, .9, .95, .96, .97, .98,.99, .993, .995,.9955, .996, .9962,.9964, .9965 ,.997, .999, .999999]
for thr in thresholds:
    pred_labels = (y_pred_prob >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels).ravel()

    tpr = tp / (tp + fn)
    fpr = fp / (fp+tn)

    print("Threshold:", thr)
    print("True Positive Rate (TPR):", tpr)
    print("False Positive Rate (FPR):", fpr)
    print("")

