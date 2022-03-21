import os
from sklearn.externals import joblib
import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np
import autosklearn.classification
from math import floor

total_time = 86400
binary=True
data = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'tested_datasets')
                                , 'set-a_4sigmas_complete_fa2_oblimin_kmeans_undersampled+BUN_GCS+categorical.csv'), delimiter=',', header=0)
n_components=len(np.unique(data['kmeans_labels']))//2 if 'kmeans_labels' in data.columns else 0

unimportant_labels = ['kmeans_labels', 'outcome', 'RecordID']
unimportant_labels.extend(['pc_'+str(i) for i in range(1,n_components+1)])

X = data.loc[:,[column for column in data.columns if column not in unimportant_labels]]
y = data.loc[:, 'outcome'] if binary else data.loc[:, 'kmeans_labels']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1, test_size=0.5)

automl = autosklearn.classification.AutoSklearnClassifier(
    include_preprocessors=['no_preprocessing']
    , include_estimators=['gaussian_nb'
                          , 'bernoulli_nb'
                          , 'gradient_boosting'
                          , 'liblinear_svc'
                          , 'random_forest']
    , ensemble_size=1
    , initial_configurations_via_metalearning=0
    , time_left_for_this_task=total_time
    , per_run_time_limit=floor(0.1*total_time)
    , ml_memory_limit=32000
    , tmp_folder='/home/flavio/Documentos/my_env/physionet_challenge/tmp/autosklearn_cv_example_tmp'
    , output_folder='/home/flavio/Documentos/my_env/physionet_challenge/tmp/tmp/autosklearn_cv_example_out'
    #, tmp_folder='/home/sarabada/Documents/physionet_challenge/tmp/autosklearn_cv_example_tmp'
    #, output_folder='/home/sarabada/Documents/physionet_challenge/tmp/tmp/autosklearn_cv_example_out'
    , delete_tmp_folder_after_terminate=True
    , resampling_strategy='cv'
    , resampling_strategy_arguments={'folds': 10}
)

automl.fit(X_train, y_train)
automl.refit(X_train.copy(), y_train.copy())
joblib.dump(automl, 'auto-sklearn_model.pkl') 
print(automl.show_models())
predictions = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
