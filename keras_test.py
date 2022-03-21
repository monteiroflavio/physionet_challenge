import re
import os
import warnings
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD
from sklearn.model_selection import GridSearchCV
from keras.metrics import categorical_crossentropy
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, classification_report
from postprocessing import binaryConfusionMatrix, binarizeLabels, oneVsAllConfusionMatrix
from sklearn.metrics import make_scorer, precision_score, recall_score, roc_curve, auc, f1_score
from keras.layers import Activation, Flatten, Dense, Conv1D, MaxPooling1D, Dropout, Embedding, LSTM

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def f1_weighted(y_true, y_pred, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return f1_score(y_true, y_pred, average='macro')

def precision_weighted(y_true, y_pred, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return precision_score(y_true, y_pred, average='macro')

def recall_weighted(y_true, y_pred, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return recall_score(y_true, y_pred, average='macro')

def precision_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return precision_score(y_true, y_pred, labels=[pos_label], average='macro')

def recall_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return recall_score(y_true, y_pred, labels=[pos_label], average='macro')

def f1_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return f1_score(y_true, y_pred, labels=[pos_label], average='macro')

def score_1_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return min(precision_score(y_true, y_pred, labels=[pos_label], average='macro')
               , recall_score(y_true, y_pred, labels=[pos_label], average='macro'))

def auc_score(y_true, y_pred, pos_label, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)
    return auc(fpr, tpr)

def tp_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    confusionMatrix = confusion_matrix(y_true, y_pred)
    if binarize:
        confusionMatrix = binaryConfusionMatrix(confusionMatrix, n_components)
    return oneVsAllConfusionMatrix(confusionMatrix, pos_label)[0][0]

def tn_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    confusionMatrix = confusion_matrix(y_true, y_pred)
    if binarize:
        confusionMatrix = binaryConfusionMatrix(confusionMatrix, n_components)
    return oneVsAllConfusionMatrix(confusionMatrix, pos_label)[1][1]

def fp_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    confusionMatrix = confusion_matrix(y_true, y_pred)
    if binarize:
        confusionMatrix = binaryConfusionMatrix(confusionMatrix,n_components)
    return oneVsAllConfusionMatrix(confusionMatrix, pos_label)[0][1]

def fn_at(y_true, y_pred, pos_label, binarize=False, n_components=0):
    confusionMatrix = confusion_matrix(y_true, y_pred)
    if binarize:
        confusionMatrix = binaryConfusionMatrix(confusionMatrix, n_components)
    return oneVsAllConfusionMatrix(confusionMatrix, pos_label)[1][0]

def error(mean, sd):
    return 1 - (((mean + sd)*(mean - sd))/(mean ** 2))

def create_scoring(y, n_components):
    scoring = {}
    if len(set(y)) > 2:
        scoring['f1-weighted_m'] = make_scorer(f1_weighted)
        scoring['precision-weighted_m'] = make_scorer(precision_weighted)
        scoring['recall-weighted_m'] = make_scorer(recall_weighted)
        scoring['f1-weighted_b'] = make_scorer(f1_weighted, binarize=True, n_components=n_components)
        scoring['precision-weighted_b'] = make_scorer(precision_weighted, binarize=True, n_components=n_components)
        scoring['recall-weighted_b'] = make_scorer(recall_weighted, binarize=True, n_components=n_components)
        labels = sorted(set(y))
        for label in labels:
            scoring['m_'+str(label)+'_f1'] = make_scorer(f1_at, pos_label=labels.index(label))
            scoring['m_'+str(label)+'_precision'] = make_scorer(precision_at, pos_label=labels.index(label))
            scoring['m_'+str(label)+'_recall'] = make_scorer(recall_at, pos_label=labels.index(label))
            scoring['m_'+str(label)+'_score1'] = make_scorer(score_1_at, pos_label=labels.index(label))
            scoring['m_'+str(label)+'_tp'] = make_scorer(tp_at, pos_label=labels.index(label))
            scoring['m_'+str(label)+'_tn'] = make_scorer(tn_at, pos_label=labels.index(label))
            scoring['m_'+str(label)+'_fp'] = make_scorer(fp_at, pos_label=labels.index(label))
            scoring['m_'+str(label)+'_fn'] = make_scorer(fn_at, pos_label=labels.index(label))
#            scoring['m_'+str(label)+'_auc_score'] = make_scorer(auc_score, pos_label=labels.index(label))
        for i in range(2):
            scoring['b_'+str(i)+'_f1'] = make_scorer(f1_at, pos_label=i, binarize=True, n_components=n_components)
            scoring['b_'+str(i)+'_precision'] = make_scorer(precision_at, pos_label=i, binarize=True, n_components=n_components)
            scoring['b_'+str(i)+'_recall'] = make_scorer(recall_at, pos_label=i, binarize=True, n_components=n_components)
            scoring['b_'+str(i)+'_score1'] = make_scorer(score_1_at, pos_label=i, binarize=True, n_components=n_components)
            scoring['b_'+str(i)+'_tp'] = make_scorer(tp_at, pos_label=i, binarize=True, n_components=n_components)
            scoring['b_'+str(i)+'_tn'] = make_scorer(tn_at, pos_label=i, binarize=True, n_components=n_components)
            scoring['b_'+str(i)+'_fp'] = make_scorer(fp_at, pos_label=i, binarize=True, n_components=n_components)
            scoring['b_'+str(i)+'_fn'] = make_scorer(fn_at, pos_label=i, binarize=True, n_components=n_components)
#            scoring['b_'+str(i)+'_auc_score'] = make_scorer(auc_score, pos_label=i, binarize=True)
    else:
        scoring['f1-weighted_b'] = make_scorer(f1_weighted)
        scoring['precision-weighted_b'] = make_scorer(precision_weighted)
        scoring['recall-weighted_b'] = make_scorer(recall_weighted)
        for i in range(2):
            scoring['b_'+str(i)+'_f1'] = make_scorer(f1_at, pos_label=i)
            scoring['b_'+str(i)+'_precision'] = make_scorer(precision_at, pos_label=i)
            scoring['b_'+str(i)+'_recall'] = make_scorer(recall_at, pos_label=i)
            scoring['b_'+str(i)+'_score1'] = make_scorer(score_1_at, pos_label=i)
            scoring['b_'+str(i)+'_tp'] = make_scorer(tp_at, pos_label=i)
            scoring['b_'+str(i)+'_tn'] = make_scorer(tn_at, pos_label=i)
            scoring['b_'+str(i)+'_fp'] = make_scorer(fp_at, pos_label=i)
            scoring['b_'+str(i)+'_fn'] = make_scorer(fn_at, pos_label=i)
#            scoring['b_'+str(i)+'_auc_score'] = make_scorer(auc_score, pos_label=i)
    return scoring

def structurate_cross_validation_result(results):
    structured_results = {}
    for key in sorted(results.keys()):
        if 'test_' in key:
            if re.match('^(test_)(m|b)_(.+_)(.*)', key):
                if key.split('_')[1] not in structured_results.keys():
                    structured_results[key.split('_')[1]] = {'labels': {}}
                if key.split('_')[2] not in structured_results[key.split('_')[1]]['labels'].keys():
                    structured_results[key.split('_')[1]]['labels'][key.split('_')[2]] = {}
                if key.split('_')[3] not in structured_results[key.split('_')[1]]['labels'][key.split('_')[2]].keys():
                    structured_results[key.split('_')[1]]['labels'][key.split('_')[2]][key.split('_')[3]] = {
                        'interactions': results[key]
                        , 'mean': np.mean(results[key])
                        , 'sd': np.std(results[key])
                    }
            elif re.match('^(test_)(.+_)(m|b)', key):
                if key.split('_')[2] not in structured_results.keys():
                    structured_results[key.split('_')[2]] = {'labels': {}}
                if key.split('_')[1] not in structured_results[key.split('_')[2]].keys():
                    structured_results[key.split('_')[2]][key.split('_')[1]] = {
                        'interactions': results[key]
                        , 'mean': np.mean(results[key])
                        , 'sd': np.std(results[key])
                    }
    return structured_results

def csv_cv_results(scores):
    result = ""
    if 'm' in scores.keys():
        result += str(scores['m']['f1-weighted']['mean'])+','+str(scores['m']['f1-weighted']['sd'])+','
        result += str(error(scores['m']['f1-weighted']['mean'], scores['m']['f1-weighted']['sd']))+','
        result += str(scores['m']['precision-weighted']['mean'])+','+str(scores['m']['precision-weighted']['sd'])+','
        result += str(scores['m']['recall-weighted']['mean'])+','+str(scores['m']['recall-weighted']['sd'])+','
        for label in scores['m']['labels'].keys():
            for score in scores['m']['labels'][label].keys():
                result += str(scores['m']['labels'][label][score]['mean'])+','+str(scores['m']['labels'][label][score]['sd'])+','
                if score == 'f1':
                    result += str(error(scores['m']['labels'][label][score]['mean'], scores['m']['labels'][label][score]['sd']))+','
    result += str(scores['b']['f1-weighted']['mean'])+','+str(scores['b']['f1-weighted']['sd'])+','
    result += str(error(scores['b']['f1-weighted']['mean'], scores['b']['f1-weighted']['sd']))+','
    result += str(scores['b']['precision-weighted']['mean'])+','+str(scores['b']['precision-weighted']['sd'])+','
    result += str(scores['b']['recall-weighted']['mean'])+','+str(scores['b']['recall-weighted']['sd'])+','
    for label in scores['b']['labels'].keys():
        for score in scores['b']['labels'][label].keys():
            result += str(scores['b']['labels'][label][score]['mean'])+','+str(scores['b']['labels'][label][score]['sd'])+','
            if score == 'f1':
                result += str(error(scores['b']['labels'][label][score]['mean'], scores['b']['labels'][label][score]['sd']))+','
    return result

def load_data(cnn=False, binary=True):
    data_a = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'tested_datasets')
                                , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_GCS+categorical.csv"), delimiter=',', header=0)
    n_components=len(np.unique(data_a['kmeans_labels']))//2 if 'kmeans_labels' in data_a.columns else 0

    if not binary:
        data_a['kmeans_labels'] = data_a['kmeans_labels'] -1

    unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
    unimportant_labels.extend(['pc_'+str(i) for i in range(1,n_components+1)])

    X_a = data_a.loc[:,[column for column in data_a.columns if column not in unimportant_labels]]
    y_a = data_a.loc[:, 'outcome'] if binary else data_a.loc[:, 'kmeans_labels']

    X_a = X_a.values
    y_a = y_a.values
    
    if cnn:
        X_a = X_a.reshape(X_a.shape[0], X_a.shape[1], 1)
    
    return X_a, y_a

def create_model(
        activation12='tanh'
        , activation3='softplus'
        , optimizer='sgd'
        , learn_rate=0.001
        , momentum=0.8
        , init_mode='he_uniform'
        , weight_constraint=5
        , dropout_rate=0.7
        , neurons = 6
        
):
    model = Sequential()
    
    ## ANN ##
    model.add(Dense(neurons, input_dim=12, kernel_initializer=init_mode, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(round(neurons)*2/3)+1, kernel_initializer=init_mode, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dense(int(round(neurons)*2/3)+1, kernel_initializer=init_mode, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dense(int(round(neurons)*2/3)+1, kernel_initializer=init_mode, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dense(int(round(neurons)*2/3)+1, kernel_initializer=init_mode, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dense(int(round(neurons)*2/3)+1, kernel_initializer=init_mode, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dense(int(round(neurons)*2/3)+1, kernel_initializer=init_mode, activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=init_mode, activation='softsign'))

    ## CNN ##
    #model.add(Conv1D(8, 1 ,activation='relu' ,input_shape=(8,1)))
    #model.add(MaxPooling1D(1))
    #model.add(Conv1D(16, 1, activation='relu'))
    #model.add(MaxPooling1D(1))
    #model.add(Flatten())
    #model.add(Dense(1000, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))

    ## LSTM ##
    #model.add(Embedding(input_dim=9, output_dim=1))
    #model.add(LSTM(8))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))

    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    ## UNCOMMENT FOR LSTM ##
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

    return model

binary=True

X_a, y_a = load_data(cnn=False)
model = KerasClassifier(build_fn=create_model, batch_size=1, epochs=500, shuffle=True, verbose=2)

## GRID TESTING SHIT ##
#batch_size =[1, 10, 50, 100]
#epochs = [10, 50, 100, 500]
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#weight_constraint = [1, 2, 3,4, 5]
#neurons = [1, 2, 4, 8, 16, 32, 64]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#param_grid = dict(
    #optimizer=optimizer
    #batch_size=batch_size
    #, epochs=epochs
    #learn_rate=learn_rate
    #, momentum=momentum
    #init_mode=init_mode
    #activation12=activation
    #, activation3=activation
    #weight_constraint=weight_constraint
    #, dropout_rate=dropout_rate
    #neurons= neurons
#)

#grid = GridSearchCV(estimator=model
#                    , param_grid=param_grid
#                    , n_jobs=-1
#                    , scoring=make_scorer(score_1_at, pos_label=1))
#grid_result = grid.fit(X_a, y_a)

#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

## CV TESTING SHIT ##
scores = structurate_cross_validation_result(
    cross_validate(model
                   , X_a
                   , y_a
                   , cv=3
                   , n_jobs=-1
                   , scoring=create_scoring(y_a, 2)
                   , return_train_score=False)
    )
print(csv_cv_results(scores))


## UNCOMMENT FOR CNN ##
#model.fit(X_a_cp, y_a.values, batch_size=1, epochs=100, shuffle=True, verbose=2)
#model.fit(X_a.values, y_a.values, batch_size=1, epochs=100, shuffle=True, verbose=2)

#data_b = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'tested_datasets')
#                                , "set-b_4sigmas_complete_fa2_oblimin_kmeans+BUN_GCS+ICUType2_MechVent.csv"), delimiter=',', header=0)
#n_components=len(np.unique(data_b['kmeans_labels']))//2 if 'kmeans_labels' in data_b.columns else 0

#if not binary:
#    data_b['kmeans_labels'] = data_b['kmeans_labels'] -1

#unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
#unimportant_labels.extend(['pc_'+str(i) for i in range(1,n_components+1)])

#X_b = data_b.loc[:,[column for column in data_b.columns if column not in unimportant_labels]]
#y_b = data_b.loc[:, 'outcome'] if binary else data_b.loc[:, 'kmeans_labels']

## UNCOMMENT FOR CNN ##
#X_b_cp = X_b.values
#X_b_cp = X_b_cp.reshape(X_b_cp.shape[0], X_b_cp.shape[1], 1)

#y_pred_b = model.predict_classes(X_b_cp)
#y_pred_b = model.predict_classes(X_b)
#confusionMatrix = normalize_confusion_matrix(confusion_matrix(y_b, y_pred_b))

#print(confusion_matrix(y_b, y_pred_b))
#print(precision_from_cm(confusionMatrix[1][1],confusionMatrix[0][1]))
#print(recall_from_cm(confusionMatrix[1][1],confusionMatrix[1][0]))

#data_c = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'tested_datasets')
#                                , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+BUN_GCS+ICUType2_MechVent.csv"), delimiter=',', header=0)
#n_components=len(np.unique(data_c['kmeans_labels']))//2 if 'kmeans_labels' in data_c.columns else 0

#if not binary:
#    data_c['kmeans_labels'] = data_c['kmeans_labels'] -1

#unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
#unimportant_labels.extend(['pc_'+str(i) for i in range(1,n_components+1)])

#X_c = data_c.loc[:,[column for column in data_c.columns if column not in unimportant_labels]]
#y_c = data_c.loc[:, 'outcome'] if binary else data_c.loc[:, 'kmeans_labels']

## UNCOMMENT FOR CNN ##
#X_c_cp = X_c.values
#X_c_cp = X_c_cp.reshape(X_c_cp.shape[0], X_c_cp.shape[1], 1)

#y_pred_c = model.predict_classes(X_c_cp)
#y_pred_c = model.predict_classes(X_c)
#confusionMatrix_2 = normalize_confusion_matrix(confusion_matrix(y_c, y_pred_c))

#print(confusion_matrix(y_c, y_pred_c))
#print(precision_from_cm(confusionMatrix_2[1][1],confusionMatrix_2[0][1]))
#print(recall_from_cm(confusionMatrix_2[1][1],confusionMatrix_2[1][0]))
