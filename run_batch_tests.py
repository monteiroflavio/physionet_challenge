import re
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from postprocessing import binarizeLabels, binarizeConfusionMatrix
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from postprocessing import binaryConfusionMatrix, binarizeLabels, oneVsAllConfusionMatrix
from sklearn.metrics import make_scorer, precision_score, recall_score, roc_curve, auc, f1_score, roc_auc_score
from pprint import PrettyPrinter

warnings.filterwarnings("ignore")
pprint = PrettyPrinter(indent=4)
datasets_path = os.path.join(os.getcwd(), 'tested_datasets')
classifiers_path = os.path.join(os.getcwd(), 'persisted_classifiers')

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

def auc_score(y_true, y_pred, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def auroc_score(y_true, y_pred, binarize=False, n_components=0):
    if binarize:
        y_true = binarizeLabels(y_true, range(n_components))
        y_pred = binarizeLabels(y_pred, range(n_components))
    return roc_auc_score(y_true, y_pred)

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
        scoring['auroc_b'] = make_scorer(auroc_score)
        scoring['auc-score_b'] = make_scorer(auc_score)
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
    if 'auroc' in scores['b'].keys():
        result += str(scores['b']['auroc']['mean'])+','+str(scores['b']['auroc']['sd'])+','
    if 'auc-score' in scores['b'].keys():
        result += str(scores['b']['auc-score']['mean'])+','+str(scores['b']['auc-score']['sd'])+','
    for label in scores['b']['labels'].keys():
        for score in scores['b']['labels'][label].keys():
            result += str(scores['b']['labels'][label][score]['mean'])+','+str(scores['b']['labels'][label][score]['sd'])+','
            if score == 'f1':
                result += str(error(scores['b']['labels'][label][score]['mean'], scores['b']['labels'][label][score]['sd']))+','
    return result

def persist_classifier(filename, classifier, binary=True, radical_name="classifier"):
    data = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'tested_datasets'), filename), delimiter=',', header=0)
    n_components=len(np.unique(data['kmeans_labels']))//2 if 'kmeans_labels' in data.columns else 0
    
    if not binary:
        data['kmeans_labels'] = data['kmeans_labels'] -1

    unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
    unimportant_labels.extend(['pc_'+str(i) for i in range(1,n_components+1)])
    
    X = data.loc[:,[column for column in data.columns if column not in unimportant_labels]]
    y = data.loc[:, 'outcome'] if binary else data.loc[:, 'kmeans_labels']
    
    classifier.fit(X, y)

    classifier_type = "b-" if binary else "m-"
    joblib.dump(classifier, os.path.join(classifiers_path, classifier_type+radical_name+"("+filename.split(".csv")[0]+").pkl"))

def test_persisted_classifier(data_filename, classifier_filename, binary=True):
    data = pd.read_csv(os.path.join(datasets_path, data_filename), delimiter=',', header=0)
    n_components=len(np.unique(data['kmeans_labels']))//2 if 'kmeans_labels' in data.columns else 0
    confusionMatrix = []
    
    if not binary:
        data['kmeans_labels'] = data['kmeans_labels'] -1

    unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
    unimportant_labels.extend(['pc_'+str(i) for i in range(1,n_components+1)])
    
    X = data.loc[:,[column for column in data.columns if column not in unimportant_labels]]
    y = data.loc[:, 'outcome'] if binary else data.loc[:, 'kmeans_labels']

    classifier = joblib.load(os.path.join(classifiers_path, classifier_filename))
    print(X.shape)
    print(list(X))
    y_pred = classifier.predict(X)
    #print(classification_report(y, y_pred))
    #print(confusion_matrix(y, y_pred))
    #print(y.name)
    if not binary:
        y_pred = binarizeLabels(y_pred, [0,1])
        #confusionMatrix = normalize_confusion_matrix(confusion_matrix(data.loc[:, "outcome"], y_pred))
        #print(confusion_matrix)
        #print(confusion_matrix(data.loc[:, "outcome"], y_pred))
    confusionMatrix = normalize_confusion_matrix(confusion_matrix(data.loc[:, "outcome"], y_pred))
    #print(confusionMatrix)
    result = ""
    result += str(f1_score(data.loc[:,"outcome"], y_pred, average='macro'))+","
    result += str(roc_auc_score(data.loc[:,"outcome"], y_pred))+","
    result += str(precision_score(data.loc[:,"outcome"], y_pred, average='macro'))+","
    result += str(recall_score(data.loc[:,"outcome"], y_pred, average='macro'))+","
    result += str(precision_from_cm(confusionMatrix[0][0],confusionMatrix[1][0]))+","
    result += str(recall_from_cm(confusionMatrix[0][0],confusionMatrix[0][1]))+","
    result += str(min(precision_from_cm(confusionMatrix[0][0],confusionMatrix[1][0])
                      , recall_from_cm(confusionMatrix[0][0],confusionMatrix[0][1])))+","
    result += str(precision_from_cm(confusionMatrix[1][1],confusionMatrix[0][1]))+","
    result += str(recall_from_cm(confusionMatrix[1][1],confusionMatrix[1][0]))+","
    result += str(min(precision_from_cm(confusionMatrix[1][1],confusionMatrix[0][1])
                  , recall_from_cm(confusionMatrix[1][1],confusionMatrix[1][0])))

    print(result)


def run_test(filename, binary, classifier):
    data = pd.read_csv(os.path.join(datasets_path, filename),
                       delimiter=',', header=0)
    n_components = len(np.unique(data['kmeans_labels'])) // 2 \
        if 'kmeans_labels' in data.columns \
        else 0

    if not binary:
        data['kmeans_labels'] = data['kmeans_labels'] - 1

    unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
    unimportant_labels.extend(['pc_' + str(i)
                               for i in range(1, n_components + 1)])

    X = data.loc[:, [column for column in data.columns
                     if column not in unimportant_labels]]
    y = data.loc[:, 'outcome'] if binary else data.loc[:, 'kmeans_labels']

    scores = structurate_cross_validation_result(
        cross_validate(classifier, X, y, cv=10, n_jobs=-1,
                       scoring=create_scoring(y, n_components),
                       return_train_score=False)
    )
    print(X.shape)
    print(list(X))
    print(y.name)
    print(csv_cv_results(scores))
    print('\n')


def test_batch(classifiers, filenames, binary=False):
    for filename in filenames:
        print(filename)
        for classifier in classifiers:
            print(classifier["classifier"])
            run_test(filename, binary, classifier["classifier"])


def persist_batch(classifiers, filenames, binary=False):
    for filename in filenames:
        print(filename)
        for classifier in classifiers:
            print(classifier["classifier"])
            persist_classifier(filename,
                               classifier["classifier"],
                               binary=binary,
                               radical_name=classifier["tag"])


def test_persisted_batch(classifiers, filenames, classifier_set, binary=False):
    classifier_type = "b-" if binary else "m-"
    for filename in filenames:
        print(filename)
        for classifier in classifiers:
            print(classifier_type+classifier["tag"]+"("+classifier_set+").pkl")
            test_persisted_classifier(filename,
                                      os.path.join(classifiers_path,
                                                   classifier_type
                                                   + classifier["tag"]
                                                   + "("+classifier_set
                                                   + ").pkl"),
                                      binary=binary)


classifiers = [
    {"classifier": RandomForestClassifier(n_estimators=1000,
                                          criterion='gini',
                                          max_features=None),
     "tag": "random_forest"},
    {"classifier": GaussianNB(), "tag": "gaussian_nb"},
    {"classifier": GradientBoostingClassifier(n_estimators=1000,
                                              learning_rate=0.1,
                                              max_features=None),
     "tag": "gradient_boosting"},
    {"classifier": SVC(kernel='linear'), "tag": "svc_linear"},
    {"classifier": MLPClassifier(hidden_layer_sizes=(100, 100),
                                 max_iter=1000),
     "tag": "mlp"},
    {"classifier": BernoulliNB(alpha=0.0000001), "tag": "bernoulli_nb"},
    {"classifier": SVC(kernel='rbf', C=10, gamma=0.1), "tag": "svc_rbf"},
    {"classifier": ExtraTreesClassifier(n_estimators=1000,
                                        max_features=None),
     "tag": "extra_trees"}
]

filenames = [
    # 'set-a_4sigmas_complete.csv'
    # 'set-a_4sigmas_complete_pca.csv'
    # "set-a_4sigmas_complete_fa1_varimax_kmeans.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Age.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Age_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Age+BUN+GCS_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Albumin_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Albumin_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+ALP_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+ALP_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Bilirubin_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Bilirubin_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+BUN_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+BUN_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+BUN+GCS_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+BUN+GCS+Na_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+BUN+GCS+Na+cat_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Creatinine_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Creatinine_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+GCS_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+GCS_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+HCT_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+HCT_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Lactate_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Lactate_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Mg_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Mg_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Na_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Na_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Platelets_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Platelets_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+SAPS.I.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+SAPS.I_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+SOFA.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+SOFA_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+SysABP_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+SysABP_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Urine_mean.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+Urine_mean_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+2best_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+3best_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+4best_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+5best_oversampled.csv"
    # "_set-a_4sigmas_complete_fa2_varimax_kmeans+5best+cat_oversampled.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans_oversampled.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Age.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Albumin_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+ALP_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Bilirubin_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+BUN_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Creatinine_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+GCS_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Glucose_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+HCT_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+K_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Lactate_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Na_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Platelets_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+SAPS.I.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+SOFA.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+SysABP_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+Urine_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+WBC_mean.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+2best.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+3best.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+4best.csv"
    # "set-a_4sigmas_complete_fa2_varimax_kmeans+5best.csv"
    # "set-a_4sigmas_complete_fa1_oblimin_kmeans.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled1.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled2.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Age.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Albumin_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+ALP_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+AST_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Bilirubin_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Creatinine_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+DiasABP_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+GCS_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Glucose_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+HR_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+K_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Lactate_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+MAP_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Mg_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+NIDiasABP_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+NIMAP_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+NISysABP_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+PaCO2_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+PaO2_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Platelets_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+SAPS.I.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+SOFA.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+SysABP_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Temp_mean.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Weight.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+2best.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+3best.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+3bestAST.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+4best.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+5best.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+5best+ICUType_MechVent.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+5best+ICUType2_MechVent.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+4best-ALP.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_GCS.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_GCS+ICUType2_MechVent.csv"
    # "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_GCS+categorical.csv"
    # , "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_GCS+ICUType_MechVent.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+BUN_mean.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+BUN_mean.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+GCS_mean.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+GCS_mean.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+2best.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+2best.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+3best.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+3best.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+3bestAST.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+3bestAST.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+4best.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+4best.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+4best-ALP.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+4best-ALP.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+5best.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+5best.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+BUN_GCS.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+BUN_GCS.csv"
    # "set-b_4sigmas_complete_fa2_varimax_kmeans.csv"
    # "set-b_4sigmas_complete_fa2_varimax_kmeans+2best.csv"
    # , "set-c_4sigmas_complete_fa2_varimax_kmeans+2best.csv"
    # "set-b_4sigmas_complete_fa2_varimax_kmeans+3best.csv"
    # , "set-c_4sigmas_complete_fa2_varimax_kmeans+3best.csv"
    # "set-b_4sigmas_complete_fa2_varimax_kmeans+4best.csv"
    # , "set-c_4sigmas_complete_fa2_varimax_kmeans+4best.csv"
    # "set-b_4sigmas_complete_fa2_varimax_kmeans+5best.csv"
    # , "set-c_4sigmas_complete_fa2_varimax_kmeans+5best.csv"
    # , "set-c_4sigmas_complete_fa2_varimax_kmeans.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+5best+ICUType_MechVent.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+5best+ICUType_MechVent.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+5best+ICUType2_MechVent.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+5best+ICUType2_MechVent.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+BUN_GCS+categorical.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+BUN_GCS+categorical.csv"
    # "set-b_4sigmas_complete_fa2_oblimin_kmeans+BUN_GCS+ICUType_MechVent.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+BUN_GCS+ICUType_MechVent.csv"
    # , "set-b_4sigmas_complete_fa2_oblimin_kmeans+BUN_GCS+ICUType2_MechVent.csv"
    # , "set-c_4sigmas_complete_fa2_oblimin_kmeans2+BUN_GCS+ICUType2_MechVent.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Age.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Albumin_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+ALP_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+ALT_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+AST_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Creatinine_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+DiasABP_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+FiO2_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+GCS_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Glucose_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+HCO3_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Height.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+HR_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+K_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Lactate_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+MAP_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Mg_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+NIDiasABP_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+NIMAP_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+NISysABP_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+PaCO2_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+PaO2_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Platelets_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+SaO2_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+SAPS.I.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+SOFA.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Urine_mean.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+Weight.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+2best.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+3best.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+4best+mca.csv"
    # , "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+4best.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+4best+ICUType2_MechVent.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+4bestSOFA.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+4bestSaO2.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+5best.csv"
    # "set-a+b_4sigmas_complete_fa2_oblimin_kmeans_undersampled+6best.csv"
    # "set-c_4sigmas_complete_fa2_oblimin(a+b)_kmeans.csv"
    # "set-c_4sigmas_complete_fa2_oblimin(a+b)_kmeans+4best.csv"
    # "set-c_4sigmas_complete_fa2_oblimin(a+b)_kmeans+4best+mca.csv"
    # "set-a+b+c_4_sigmas_complete.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+ALP_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+ALT_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+AST_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Bilirubin_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+BUN_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Creatinine_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+DiasABP_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+FiO2_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+GCS_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Glucose_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+HCO3_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+HR_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Lactate_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+MAP_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Na_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+NIDiasABP_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+NIMAP_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+NISysABP_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+PaCO2_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+PaO2_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Platelets_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+SaO2_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+SAPS.I.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+SOFA.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Temp_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Urine_mean.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+Weight.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+2best.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_undersampled_kmeans+3best.csv"
    # "set-a+b+c_4sigmas_complete_fa2_oblimin_kmeans_undersampled(a).csv"
    # "set-a_4sigmas+Age_undersampled.csv"
    # "set-b_4sigmas+Age.csv"
    # , "set-c_4sigmas+Age.csv"
    # "set-a_4sigmas+Albumin_mean_undersampled.csv"
    # "set-a_4sigmas+ALP_mean_undersampled.csv"
    # "set-a_4sigmas+ALT_mean_undersampled.csv"
    # "set-a_4sigmas+AST_mean_undersampled.csv"
    # "set-a_4sigmas+Bilirubin_mean_undersampled.csv"
    # , "set-a_4sigmas+BUN_mean_undersampled.csv"
    # "set-b_4sigmas+BUN.csv"
    # , "set-c_4sigmas+BUN.csv"
    # , "set-a_4sigmas+Creatinine_mean_undersampled.csv"
    # "set-b_4sigmas+Creatinine.csv"
    # , "set-c_4sigmas+Creatinine.csv"
    # "set-a_4sigmas+DiasABP_mean_undersampled.csv"
    # "set-a_4sigmas+FiO2_mean_undersampled.csv"
    # , "set-a_4sigmas+GCS_mean_undersampled.csv"
    # "set-b_4sigmas+GCS.csv"
    # , "set-c_4sigmas+GCS.csv"
    # "set-a_4sigmas+Glucose_mean_undersampled.csv"
    # , "set-a_4sigmas+HCO3_mean_undersampled.csv"
    # "set-b_4sigmas+HCO3.csv"
    # , "set-c_4sigmas+HCO3.csv"
    # "set-a_4sigmas+HR_mean_undersampled.csv"
    # , "set-a_4sigmas+SysABP_mean_undersampled.csv"
    # "set-b_4sigmas+SysABP.csv"
    # , "set-c_4sigmas+SysABP.csv"
    # "set-a_4sigmas+Lactate_mean_undersampled.csv"
    # "set-a_4sigmas+Mg_mean_undersampled.csv"
    # "set-a_4sigmas+NIDiasABP_mean_undersampled.csv"
    # "set-a_4sigmas+NIMAP_mean_undersampled.csv"
    # "set-a_4sigmas+NISysABP_mean_undersampled.csv"
    # "set-a_4sigmas+PaCO2_mean_undersampled.csv"
    # "set-a_4sigmas+PaO2_mean_undersampled.csv"
    # "set-a_4sigmas+Platelets_mean_undersampled.csv"
    # "set-a_4sigmas+RespRate_mean_undersampled.csv"
    # "set-a_4sigmas+SaO2_mean_undersampled.csv"
    # "set-a_4sigmas+Temp_mean_undersampled.csv"
    # , "set-a_4sigmas+Urine_mean_undersampled.csv"
    # "set-b_4sigmas+Urine.csv"
    # , "set-c_4sigmas+Urine.csv"
    # "set-a_4sigmas+Weight_undersampled.csv"
    # "set-a_4sigmas+2best_undersampled.csv"
    # "set-b_4sigmas+2best.csv"
    # , "set-c_4sigmas+2best.csv"
    # , "set-a_4sigmas+3best_undersampled.csv"
    # "set-b_4sigmas+3best.csv"
    # , "set-c_4sigmas+3best.csv"
    # , "set-a_4sigmas+4best_undersampled.csv"
    # "set-b_4sigmas+4best.csv"
    # , "set-c_4sigmas+4best.csv"
    # , "set-a_4sigmas+5best_undersampled.csv"
    # "set-b_4sigmas+5best.csv"
    # , "set-c_4sigmas+5best.csv"
    # , "set-a_4sigmas+6best_undersampled.csv"
    # "set-b_4sigmas+6best.csv"
    # , "set-c_4sigmas+6best.csv"
    # , "set-a_4sigmas+6best-Age_undersampled.csv"
    # "set-b_4sigmas+6best-Age.csv"
    # , "set-c_4sigmas+6best-Age.csv"
    # "set-a_4sigmas+6best-PaCO2_undersampled.csv"
    # "set-b_4sigmas+6best-PaCO2.csv"
    # , "set-c_4sigmas+6best-PaCO2.csv"
    # , "set-a_4sigmas+6best-PaCO2+HCO3_undersampled.csv"
    # "set-b_4sigmas+6best-PaCO2+HCO3.csv"
    # , "set-c_4sigmas+6best-PaCO2+HCO3.csv"
    # , "set-a_4sigmas+6best-PaCO2+SysABP_undersampled.csv"
    # "set-b_4sigmas+6best-PaCO2+SysABP.csv"
    # , "set-c_4sigmas+6best-PaCO2+SysABP.csv"
    # , "set-a_4sigmas+6best-PaCO2+cat_undersampled.csv"
    # "set-b_4sigmas+6best-PaCO2+cat.csv"
    # , "set-c_4sigmas+6best-PaCO2+cat.csv"
    # "set-a_4sigmas+BUN_GCS_undersampled.csv"
    # "set-b_4sigmas+BUN_GCS.csv"
    # , "set-c_4sigmas+BUN_GCS.csv"
    # "set-a_4sigmas+BUN_GCS+ca_undersampled.csv"
    # "set-b_4sigmas+BUN_GCS+ca.csv"
    # , "set-c_4sigmas+BUN_GCS+ca.csv"
    # "set-a_4sigmas+BUN_GCS_Urine_undersampled.csv"
    # "set-b_4sigmas+BUN_GCS_Urine.csv"
    # , "set-c_4sigmas+BUN_GCS_Urine.csv"
    # "set-a_4sigmas+BUN_GCS_HCO3_undersampled.csv"
    # "set-b_4sigmas+BUN_GCS_HCO3.csv"
    # , "set-c_4sigmas+BUN_GCS_HCO3.csv"
    # "set-a_4sigmas+BUN_GCS_SysABP_undersampled.csv"
    # "set-b_4sigmas+BUN_GCS_SysABP.csv"
    # , "set-c_4sigmas+BUN_GCS_SysABP.csv"
    # "set-a_4sigmas+BUN_GCS_Age_undersampled.csv"
    # "set-b_4sigmas+BUN_GCS_Age.csv"
    # , "set-c_4sigmas+BUN_GCS_Age.csv"
    # "set-a_4sigmas+BUN_GCS_Na_undersampled.csv"
    # "set-c_4sigmas+BUN_GCS_Na.csv"
    # , "set-b_4sigmas+BUN_GCS_Na.csv"
    # "set-a_4sigmas+BUN_GCS_HCT_undersampled.csv"
    # "set-b_4sigmas+BUN_GCS_HCT.csv"
    # , "set-c_4sigmas+BUN_GCS_HCT.csv"
    "_set-b_4sigmas_complete_fa2_varimax+5best+cat.csv",
    "_set-c_4sigmas_complete_fa2_varimax+5best+cat.csv"
]

# test_batch(classifiers, filenames, binary=True)
# test_batch(classifiers, filenames, binary=False)

# persist_batch(classifiers, filenames, binary=True)
# persist_batch(classifiers, filenames, binary=False)

test_persisted_batch(classifiers,
                     filenames,
                     "_set-a_4sigmas_complete_fa2_varimax_kmeans+5best+cat_oversampled",
                     binary=True)
# test_persisted_batch(classifiers, filenames, "set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_GCS+ICUType_MechVent", binary=False)
