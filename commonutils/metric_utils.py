from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.classification import brier_score_loss, matthews_corrcoef, jaccard_score, cohen_kappa_score
from sklearn.metrics.regression import explained_variance_score, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.metrics.regression import median_absolute_error, mean_squared_error, max_error
from sklearn.metrics.cluster import silhouette_score, calinski_harabasz_score, v_measure_score, fowlkes_mallows_score
from sklearn.metrics.cluster import davies_bouldin_score, homogeneity_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import completeness_score


def get_clustering_metrics(train_data, cluster_labels, ground_truth_labels=None):
    clustering_metric_dict = dict({})
    clustering_metric_dict['silhouette_score'] = silhouette_score(train_data, cluster_labels, random_state=42)
    clustering_metric_dict['calinski_harabasz_score'] = calinski_harabasz_score(train_data, cluster_labels)
    clustering_metric_dict['davies_bouldin_score'] = davies_bouldin_score(train_data, cluster_labels)

    if ground_truth_labels is not None:
        clustering_metric_dict['v_measure_score'] = v_measure_score(ground_truth_labels, cluster_labels)
        clustering_metric_dict['fowlkes_mallows_score'] = fowlkes_mallows_score(ground_truth_labels, cluster_labels)
        clustering_metric_dict['homogeneity_score'] = homogeneity_score(ground_truth_labels, cluster_labels)
        clustering_metric_dict['normalized_mutual_info_score'] = normalized_mutual_info_score(ground_truth_labels,
                                                                                              cluster_labels)
        clustering_metric_dict['adjusted_rand_score'] = adjusted_rand_score(ground_truth_labels, cluster_labels)
        clustering_metric_dict['completeness_score'] = completeness_score(ground_truth_labels, cluster_labels)

    return clustering_metric_dict


def get_classification_metrics(ground_truth_labels, predicted_labels):
    classification_metric_dict = dict({})
    classification_metric_dict['accuracy'] = accuracy_score(ground_truth_labels, predicted_labels)
    classification_metric_dict['precision'] = precision_score(ground_truth_labels, predicted_labels, average='weighted')
    classification_metric_dict['recall'] = recall_score(ground_truth_labels, predicted_labels, average='weighted')
    classification_metric_dict['f1_score'] = f1_score(ground_truth_labels, predicted_labels, average='weighted')
    classification_metric_dict['brier_score_loss'] = brier_score_loss(ground_truth_labels, predicted_labels)
    classification_metric_dict['matthews_corr_coef'] = matthews_corrcoef(ground_truth_labels, predicted_labels)
    classification_metric_dict['jaccard_score'] = jaccard_score(ground_truth_labels,predicted_labels, average='weighted')
    classification_metric_dict['cohen_kappa_score'] = cohen_kappa_score(ground_truth_labels, predicted_labels)

    return classification_metric_dict


def get_regression_metrics(ground_truth_value, predicted_value):
    regression_metric_dict = dict({})
    regression_metric_dict['r2_score'] = r2_score(ground_truth_value, predicted_value)
    regression_metric_dict['mean_squared_error'] = mean_squared_error(ground_truth_value, predicted_value)
    regression_metric_dict['mean_squared_log_error'] = mean_squared_log_error(ground_truth_value, predicted_value)
    regression_metric_dict['mean_absolute_error'] = mean_absolute_error(ground_truth_value, predicted_value)
    regression_metric_dict['explained_variance_score'] = explained_variance_score(ground_truth_value, predicted_value)
    regression_metric_dict['median_absolute_error'] = median_absolute_error(ground_truth_value, predicted_value)
    regression_metric_dict['max_error'] = max_error(ground_truth_value, predicted_value)

    return regression_metric_dict
