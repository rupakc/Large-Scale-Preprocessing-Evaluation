from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression,ARDRegression, PassiveAggressiveRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, TheilSenRegressor, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR


def get_gaussian_process_regressor():
    gpr = GaussianProcessRegressor(random_state=42)
    return ['GaussianProcess'], [gpr]


def get_neural_network_regressor(n_hidden_units=100):
    mlp = MLPRegressor(hidden_layer_sizes=(n_hidden_units,))
    return ['MultilayerPerceptron'], [mlp]


def get_knn_regressor(k=5):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn_model_name = 'KNN_' + str(k)
    return [knn_model_name], [knn]


def get_ensemble_regressors():
    extra = ExtraTreesRegressor(n_estimators=51,random_state=42)
    rf = RandomForestRegressor(n_estimators=51, random_state=42)
    grad = GradientBoostingRegressor(random_state=42)
    ada = AdaBoostRegressor(random_state=42)
    bag = BaggingRegressor(n_estimators=51, random_state=42)
    return ['ExtraTrees','RandomForest','GradientBoost','AdaBoost','Bagging'], [extra, rf, grad, ada, bag]


def get_linear_regressors():
    elastic = ElasticNet(random_state=42)
    sgd = SGDRegressor(random_state=42)
    linear = LinearRegression()
    ard = ARDRegression()
    passive = PassiveAggressiveRegressor(random_state=42)
    ridge = Ridge(random_state=42)
    lasso = Lasso(random_state=42)
    huber = HuberRegressor()
    theilsen = TheilSenRegressor(random_state=42)
    bayesian_ridge = BayesianRidge()
    return ['ElasticNet','SGD','Linear','ARD','PassiveAggressive','Ridge','Lasso','Huber','TheilSen','Bayesian'],\
           [elastic, sgd, linear, ard, passive, ridge, lasso, huber, theilsen, bayesian_ridge]


def get_support_vector_regression(kernel='rbf'):
    svr = SVR(kernel=kernel)
    svr_model_name = 'SVM_' + kernel
    return [svr_model_name],[svr]


def get_all_regressors(n_hidden_units=100, k=5, kernel='rbf'):
    classifier_name_list, classifier_list = get_linear_regressors()
    temp_classifier_name_list, temp_classifier_list = get_support_vector_regression(kernel=kernel)
    classifier_name_list.extend(temp_classifier_name_list)
    classifier_list.extend(temp_classifier_list)
    temp_classifier_name_list, temp_classifier_list = get_ensemble_regressors()
    classifier_name_list.extend(temp_classifier_name_list)
    classifier_list.extend(temp_classifier_list)
    temp_classifier_name_list, temp_classifier_list = get_knn_regressor(k=k)
    classifier_name_list.extend(temp_classifier_name_list)
    classifier_list.extend(classifier_list)
    temp_classifier_name_list, temp_classifier_list = get_neural_network_regressor(n_hidden_units=n_hidden_units)
    classifier_name_list.extend(temp_classifier_name_list)
    classifier_list.extend(temp_classifier_list)
    temp_classifier_name_list, temp_classifier_list = get_gaussian_process_regressor()
    classifier_name_list.extend(temp_classifier_name_list)
    classifier_list.extend(temp_classifier_list)
    return classifier_name_list, classifier_list
