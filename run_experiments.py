import test
import os
import pandas as pd
import json
import datetime
from dataframe_manipulating import slice_dataset
from classifiers import instantiateClassifier

def run_experiment(classifiers, feature_sets, test_size, is_in_hd=False, do_kmeans=False):
    if not os.path.exists(os.path.join(os.getcwd(), 'results')):
        os.makedirs(os.path.join(os.getcwd(), 'results'))

    experiment_time = str(datetime.datetime.now())
    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'results'), experiment_time)):
        os.makedirs(os.path.join(os.path.join(os.getcwd(), 'results'), experiment_time))

    if is_in_hd:
        for i in range(len(feature_sets)):
            feature_sets[i] = {'data': pd.read_csv(os.path.join(
                os.path.join(os.getcwd(), 'feature_sets')
                , feature_sets[i]), header=0, delimiter=r"\s+"),
                'name': feature_sets[i]}
        
    for feature_set in feature_sets:
        file = open(os.path.join(os.path.join(os.path.join(os.getcwd()
                                    , 'results'), experiment_time)
                        , os.path.splitext(feature_set['name'])[0])+'.txt', 'w')
        file.write('feature set: '+feature_set['name']+'\n')
        file.write('train/test proportion: '+str(1-test_size)+'/'+str(test_size)+'\n')
        file.write('number of features: '+str(len(list(feature_set['data'].iloc[:,0:-1])))+'\n')
        file.write('features used: '+'\n\t'.join(list(feature_set['data'].iloc[:,0:-1]))+'\n')
        file.write('number of samples: '+str(len(feature_set['data'].iloc[:,0:-1].index))+'\n')
        for classifier_name in classifiers:
            classifier = instantiateClassifier(classifier_name)
            file.write('----------------------------------------------------\n')
            file.write('classifier: '+classifier_name+'\n')
            file.write(json.dumps(classifier.get_params())+'\n\n')
            file.write(test.evaluate_test(classifier, feature_set['data'], kfolds=10)+'\n')
            file.write('----------------------------------------------------\n')
        file.close()

classifiers = ['lr', 'nn_1', 'nn_2', 'rf', 'et', 'svm_1', 'svm_2', 'svm_3', 'svm_4']
#feature_sets = [f for f in os.listdir(os.path.join(os.getcwd(), 'feature_sets'))
#                if os.path.isfile(os.path.join(os.path.join(os.getcwd(), 'feature_sets'), f))]

run_experiment(classifiers, feature_sets, 0.3, is_in_hd=True)
