from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC

def instantiateClassifier(classifier):
	return {
		'lr': LogisticRegression(random_state=0)
		, 'nn_1': MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100, 100), max_iter=1000, alpha=0.00001, random_state=10)
		, 'nn_2': MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100), max_iter=1000, alpha=0.00001, random_state=10)
		, 'rf': RandomForestClassifier(max_depth=200, random_state=10, criterion='entropy')
                , 'et': ExtraTreesClassifier(max_depth=200, random_state=10, n_estimators=200, criterion='entropy')
		, 'svm_1': SVC(C=5, random_state=10, decision_function_shape ="ovr", kernel='rbf')
                , 'svm_2': SVC(C=5, random_state=10, decision_function_shape ="ovr", kernel='poly')
                , 'svm_3': SVC(C=5, random_state=10, decision_function_shape ="ovr", kernel='sigmoid')
		, 'svm_4': LinearSVC(C=5, random_state=10)
	}[classifier]
