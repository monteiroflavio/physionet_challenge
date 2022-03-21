import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, roc_curve, auc, accuracy_score, f1_score
from classifiers import instantiateClassifier
from postprocessing import binarizeLabels, binarizeConfusionMatrix
import re
from pprint import PrettyPrinter
from dataframe_manipulating import slice_dataset

def f1_macro_score(y_true, y_pred, binarize=False):
        if binarize:
                y_true = binarizeLabels(y_true, [0,1,2,3])
                y_pred = binarizeLabels(y_pred, [0,1,2,3])
        return f1_score(y_true, y_pred, average='macro')
def accuracy_macro_score(y_true, y_pred, binarize=False):
        if binarize:
                y_true = binarizeLabels(y_true, [0,1,2,3])
                y_pred = binarizeLabels(y_pred, [0,1,2,3])
        return accuracy_score(y_true, y_pred)
def precision_at(y_true, y_pred, pos_label, binarize=False):
        if binarize:
                y_true = binarizeLabels(y_true, [0,1,2,3])
                y_pred = binarizeLabels(y_pred, [0,1,2,3])
        return precision_score(y_true, y_pred, labels=[pos_label], average='micro')
def recall_at(y_true, y_pred, pos_label, binarize=False):
        if binarize:
                y_true = binarizeLabels(y_true, [0,1,2,3])
                y_pred = binarizeLabels(y_pred, [0,1,2,3])
        return recall_score(y_true, y_pred, labels=[pos_label], average='micro')
def f1_at(y_true, y_pred, pos_label, binarize=False):
        if binarize:
                y_true = binarizeLabels(y_true, [0,1,2,3])
                y_pred = binarizeLabels(y_pred, [0,1,2,3])
        return f1_score(y_true, y_pred, labels=[pos_label], average='micro')
def score_1_at(y_true, y_pred, pos_label, binarize=False):
        if binarize:
                y_true = binarizeLabels(y_true, [0,1,2,3])
                y_pred = binarizeLabels(y_pred, [0,1,2,3])
        return min(precision_score(y_true, y_pred, labels=[pos_label], average='micro')
                   , recall_score(y_true, y_pred, labels=[pos_label], average='micro'))
def auc_score(y_true, y_pred, pos_label, binarize=False):
        if binarize:
                y_true = binarizeLabels(y_true, [0,1,2,3])
                y_pred = binarizeLabels(y_pred, [0,1,2,3])
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)
        return auc(fpr, tpr)
def create_scoring(y):
        scoring = {}
        if len(set(y)) > 2:
                scoring['f1-macro_m'] = make_scorer(f1_macro_score)
                scoring['accuracy_m'] = make_scorer(accuracy_macro_score)
                scoring['f1-macro_b'] = make_scorer(f1_macro_score, binarize=True)
                scoring['accuracy_b'] = make_scorer(accuracy_macro_score, binarize=True)
                labels = sorted(set(y))
                for label in labels:
                        scoring['m_'+str(label)+'_precision'] = make_scorer(precision_at, pos_label=labels.index(label))
                        scoring['m_'+str(label)+'_recall'] = make_scorer(recall_at, pos_label=labels.index(label))
                        scoring['m_'+str(label)+'_score-1'] = make_scorer(score_1_at, pos_label=labels.index(label))
                        scoring['m_'+str(label)+'_auc-score'] = make_scorer(auc_score, pos_label=labels.index(label))
                for i in range(2):
                        scoring['b_'+str(i)+'_precision'] = make_scorer(precision_at, pos_label=i, binarize=True)
                        scoring['b_'+str(i)+'_recall'] = make_scorer(recall_at, pos_label=i, binarize=True)
                        scoring['b_'+str(i)+'_score-1'] = make_scorer(score_1_at, pos_label=i, binarize=True)
                        scoring['b_'+str(i)+'_auc-score'] = make_scorer(auc_score, pos_label=i, binarize=True)
        else:
                scoring['f1-macro_b'] = make_scorer(f1_macro_score)
                scoring['accuracy_b'] = make_scorer(accuracy_macro_score)
                for i in range(2):
                        scoring['b_'+str(i)+'_precision'] = make_scorer(precision_at, pos_label=i)
                        scoring['b_'+str(i)+'_recall'] = make_scorer(recall_at, pos_label=i)
                        scoring['b_'+str(i)+'_score-1'] = make_scorer(score_1_at, pos_label=i)
                        scoring['b_'+str(i)+'_auc-score'] = make_scorer(auc_score, pos_label=i)
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

def evaluate_test(classifier, data, kfolds=3
                  , display_iterations=False):
        experiment_result = ''
        X = data.iloc[:,0:-1]
        y = data.iloc[:,-1]
        result = structurate_cross_validation_result(
                cross_validate(classifier, X, y, cv=kfolds, scoring=create_scoring(y)
                               , return_train_score=False))
        if 'm' in result.keys():
                experiment_result += 'Multi-class results over '+str(kfolds)+'-folds cross-validation:\n'
                experiment_result += 'Accuracy: '+str(result['m']['accuracy']['mean'])+', sd: '+str(result['m']['accuracy']['sd'])+'\n'
                experiment_result += 'F1-macro: '+str(result['m']['f1-macro']['mean'])+', sd: '+str(result['m']['f1-macro']['sd'])+'\n\n'
                experiment_result += 'Micro statistics for each class:\n'
                experiment_result += '{:>20}{:>20}{:>20}{:>20}\n'.format('precision', 'recall', 'score-1', 'auc-score')
                for label in result['m']['labels'].keys():
                        experiment_result += '{:}'.format(label)
                        for score in result['m']['labels'][label].keys():
                                experiment_result += '{:>20}'.format(str(np.around(result['m']['labels'][label][score]['mean'], decimals=2).item())+' '+str(np.around(result['m']['labels'][label][score]['sd'], decimals=2).item()))
                        experiment_result += '\n'
                experiment_result += '\n'
        experiment_result += 'Binary results over '+str(kfolds)+'-folds cross-validation:\n'
        experiment_result += 'Accuracy: '+str(result['b']['accuracy']['mean'])+', sd: '+str(result['b']['accuracy']['sd'])+'\n'
        experiment_result += 'F1-macro: '+str(result['b']['f1-macro']['mean'])+', sd: '+str(result['b']['f1-macro']['sd'])+'\n\n'
        experiment_result += 'Micro statistics for each class:\n'
        experiment_result += '{:>20}{:>20}{:>20}{:>20}\n'.format('precision', 'recall', 'score-1', 'auc-score')
        for label in result['b']['labels'].keys():
                experiment_result += '{:}'.format(label)
                for score in result['b']['labels'][label].keys():
                        experiment_result += '{:>20}'.format(str(np.around(result['b']['labels'][label][score]['mean'], decimals=2).item())+' '+str(np.around(result['b']['labels'][label][score]['sd'], decimals=2).item()))
                experiment_result += '\n'
        return experiment_result
