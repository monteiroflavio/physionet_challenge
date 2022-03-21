import os
import re
import test
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.externals import joblib
from postprocessing import binaryConfusionMatrix, binarizeLabels
from sklearn.metrics import accuracy_score, confusion_matrix

def normalize_confusion_matrix(confusion_matrix):
    resultant_cm = [[0,0],[0,0]]
    n_class0 = confusion_matrix[0][0]+confusion_matrix[0][1]
    n_class1 = confusion_matrix[1][0]+confusion_matrix[1][1]
    
    resultant_cm[0][0] = confusion_matrix[0][0]/n_class0/2
    resultant_cm[0][1] = 0.5 - resultant_cm[0][0]
    resultant_cm[1][0] = confusion_matrix[1][0]/n_class1/2
    resultant_cm[1][1] = 0.5 - resultant_cm[1][0]
    return resultant_cm

def precision_from_cm(tp, fp):
    return tp/(tp+fp)

def recall_from_cm(tp, fn):
    return tp/(tp+fn)

binary=True
data = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'tested_datasets')
                                                                , 'set-c_4sigmas_complete_fa2_oblimin_kmeans2+BUN_GCS+categorical.csv'), delimiter=',', header=0)
n_components=len(np.unique(data['kmeans_labels']))//2 if 'kmeans_labels' in data.columns else 0

unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
unimportant_labels.extend(['pc_'+str(i) for i in range(1,n_components+1)])

X = data.loc[:,[column for column in data.columns if column not in unimportant_labels]]
y = data.loc[:, 'outcome'] if binary else data.loc[:, 'kmeans_labels']

automl = joblib.load('auto-sklearn_model.pkl')
y_pred = automl.predict(X)

print(automl.show_models())
print(automl.sprint_statistics())
print(X.shape)
print(list(X))

if not binary:
    y_pred = binarizeLabels(y_pred, [0,1])

confusionMatrix = normalize_confusion_matrix(confusion_matrix(data.loc[:, "outcome"], y_pred))
print(precision_from_cm(confusionMatrix[0][0],confusionMatrix[1][0]))
print(recall_from_cm(confusionMatrix[0][0],confusionMatrix[0][1]))
print(confusionMatrix)
#print(test.evaluate_test(automl, data, kfolds=10))
