'''
Steps in the pipeline to be executed sequentially -
1. Load the dataframe
2. Load the metadata
3. Calculate summary statistics of the data
4. Apply pre-processing techniques on the data
5. Apply ML models on the data
6. Evaluate the model performance
7. Persist the results
'''

from commonutils import data_load_utils, metric_utils, preprocess_utils
from models import classifier_models, regressor_models, cluster_models
from constants import model_constants


class EvaluationPipeline:
    def __init__(self, filepath, type_of_model='classification'):
        self.filepath = filepath
        self.type_of_model = type_of_model

    def get_data_and_metadata(self):
        dataframe = data_load_utils.get_dataframe_from_file_name(filepath=self.filepath)
        # TODO - Get metadata from the file
        return dataframe

    @staticmethod
    def calculate_summary_statistics(dataframe):
        pass

    @staticmethod
    def get_processed_dataframe(dataframe, columns_to_impute, columns_to_encode, columns_to_scale,
                                columns_to_transform, type_of_imputation ,type_of_encoding, type_of_scaling, type_of_transformation,
                                function_transform=None, inverse_function=None,power_degree=2):
        dataframe = preprocess_utils.get_imputed_data(dataframe, columns_to_impute, type_of_imputation=type_of_imputation)
        dataframe = preprocess_utils.get_encoded_data(dataframe, columns_to_encode, type_of_encoding=type_of_encoding)
        dataframe = preprocess_utils.get_scaled_data(dataframe, columns_to_scale, type_of_scaling=type_of_scaling)
        dataframe = preprocess_utils.get_transformed_data(dataframe, columns_to_transform,
                                                          type_of_transformation=type_of_transformation,
                                                          function_transform=function_transform,
                                                          inverse_function=inverse_function, power_degree=power_degree)
        return dataframe

    def get_machine_learning_model(self):
        if self.type_of_model.lower() == model_constants.CLASSIFICATION_TYPE:
            return classifier_models.get_all_classifiers()
        elif self.type_of_model.lower() == model_constants.REGRESSION_TYPE:
            return regressor_models.get_all_regressors()
        else:
            return cluster_models.get_clustering_algorithm_list()

    def evaluate_model(self,model_list, model_name_list, X_train, X_test, y_train, y_test):
        metric_dict_list = list([])
        if self.type_of_model.lower() == model_constants.CLASSIFICATION_TYPE or self.type_of_model.lower() == model_constants.REGRESSION_TYPE:
            for model, model_name in zip(model_list, model_name_list):
                model.fit(X_train, y_train)
                predicted_values = model.predict(X_test)
                if self.type_of_model.lower() == model_constants.CLASSIFICATION_TYPE:
                    metric_dict = metric_utils.get_classification_metrics(y_test, predicted_values)
                    metric_dict_list.append(metric_dict) # TODO - Make a dictionary out of this
                else:
                    metric_dict = metric_utils.get_regression_metrics(y_test, predicted_values)
                    metric_dict_list.append(metric_dict)
        elif self.type_of_model.lower() == model_constants.CLUSTER_TYPE:
            for model_name in model_name_list:
                cluster_labels, _ = cluster_models.get_clustered_data(X_train,clustering_algorithm=model_name)
                metric_dict = metric_utils.get_clustering_metrics(X_train, cluster_labels)
                metric_dict_list.append(metric_dict)
        return metric_dict_list

    def persist_evaluation_result(self):
        pass # TODO - Write code for persisting the results

    def execute_pipeline(self):
        pass