from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


def get_gaussian_process_classifier():
    gpc = GaussianProcessClassifier(random_state=42)
    return ['GaussianProcess'], [gpc]


def get_mlp_classifier(n_hidden_units=100):
    mlp = MLPClassifier(hidden_layer_sizes=(n_hidden_units,))
    return ['MLP'], [mlp]


def get_naive_bayes_classifiers():
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    return ['GaussianNB','MultinomialNB','BernoulliNB'], [gnb, mnb, bnb]


def get_discriminant_classifiers():
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()
    return ['LDA','QDA'], [lda, qda]


def get_knn_classifier(n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model_name = 'KNN_' + str(n_neighbors)
    return [knn_model_name], [knn]


def get_ensemble_tree_classifiers():
    rf = RandomForestClassifier(n_estimators=51,random_state=42)
    grad = GradientBoostingClassifier(random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    extra = ExtraTreesClassifier(n_estimators=51, random_state=42)
    bag = BaggingClassifier(n_estimators=51, random_state=42)
    classifier_list = [rf, grad, ada, extra, bag]
    classifier_name_list = ["Random Forest","GradientBoost", "AdaBoost", 'ExtraTrees','Bagging']
    return classifier_name_list, classifier_list


def get_support_vector_classifiers(kernel='rbf'):
    support_vector = SVC(random_state=42, kernel=kernel)
    svm_model_name = 'SVM_' + kernel
    return [svm_model_name], [support_vector]


def get_linear_classifiers():
    logit_reg = LogisticRegression(random_state=42)
    passive_aggressive = PassiveAggressiveClassifier(random_state=42)
    ridge = RidgeClassifier(random_state=42)
    return ["Logistic Regression", "PassiveAggressive", "Ridge"],[logit_reg, passive_aggressive, ridge]


def get_sgd_classifier():
    sgd = SGDClassifier(random_state=42)
    return ["Stochastic Gradient Descent"] , [sgd]


def get_warm_start_classifiers():
    rf = RandomForestClassifier(n_estimators=51,random_state=42,warm_start=True)
    grad = GradientBoostingClassifier(random_state=42,warm_start=True)
    lr = LogisticRegression(solver='sag',warm_start=True,random_state=42)
    return ['Random Forest', 'Logistic Regression', 'GradientBoost'] , [rf, lr, grad]


def get_random_forest_classifier():
    rf = RandomForestClassifier(n_estimators=51, random_state=42)
    return ['Random Forest'], [rf]


def get_ada_boost_classifier():
    ada = AdaBoostClassifier(random_state=42)
    return ['AdaBoost'], [ada]


def get_grad_boost_classifier():
    grad = GradientBoostingClassifier(random_state=42, warm_start=True)
    return ['GradientBoost'], [grad]


def get_all_classifiers():
    classifier_list, classifier_name_list = get_ensemble_tree_classifiers()
    temp_classifier_list, temp_classifier_name_list = get_support_vector_classifiers()
    classifier_list.extend(temp_classifier_list)
    classifier_name_list.extend(temp_classifier_name_list)
    temp_classifier_list, temp_classifier_name_list = get_linear_classifiers()
    classifier_list.extend(temp_classifier_list)
    classifier_name_list.extend(temp_classifier_name_list)
    return classifier_name_list , classifier_list
