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
from commonutils import split_data, merge_utils, dbutils, logutils
import numpy as np

LOGGER = logutils.get_logger('Evaluation Pipeline')


class EvaluationPipeline:
    def __init__(self, filepath, type_of_model='classification'):
        self.filepath = filepath
        self.type_of_model = type_of_model

    def get_data_and_metadata(self):
        dataframe = data_load_utils.get_dataframe_from_file_name(filepath=self.filepath)
        metadata_filepath = data_load_utils.get_meta_data_filepath(filepath=self.filepath)
        metadata = data_load_utils.get_json_content(metadata_filepath)
        return dataframe, metadata

    def calculate_summary_statistics(self, dataframe):
        num_rows = len(dataframe)
        num_columns = len(dataframe.columns)
        data_filename = self.filepath.rsplit('\\',1)[1].replace(model_constants.DATA_FILE_EXTENSION, '')
        return {'num_rows': num_rows, 'num_columns': num_columns, 'data_filename': data_filename}

    @staticmethod
    def get_processed_dataframe(dataframe, columns_to_impute, columns_to_encode, columns_to_scale,
                                columns_to_transform, type_of_imputation ,type_of_encoding, type_of_scaling, type_of_transformation,
                                function_transform=None, inverse_function=None,power_degree=2, default_missing_value=np.nan):
        dataframe, _ = preprocess_utils.get_imputed_data(dataframe, columns_to_impute, type_of_imputation=type_of_imputation, missing_values=default_missing_value)
        dataframe, _ = preprocess_utils.get_encoded_data(dataframe, columns_to_encode, type_of_encoding=type_of_encoding)
        dataframe, _ = preprocess_utils.get_scaled_data(dataframe, columns_to_scale, type_of_scaling=type_of_scaling)
        dataframe, _ = preprocess_utils.get_transformed_data(dataframe, columns_to_transform,
                                                          type_of_transformation=type_of_transformation,
                                                          function_transform=function_transform,
                                                          inverse_function=inverse_function, power_degree=power_degree)
        return dataframe

    def get_machine_learning_models(self):
        if self.type_of_model.lower() == model_constants.CLASSIFICATION_TYPE:
            return classifier_models.get_all_classifiers()
        elif self.type_of_model.lower() == model_constants.REGRESSION_TYPE:
            return regressor_models.get_all_regressors()
        else:
            return cluster_models.get_clustering_algorithm_list()

    def evaluate_model(self, model_list, model_name_list, X_train, X_test, y_train, y_test):
        master_metric_dict = dict({})
        if self.type_of_model.lower() == model_constants.CLASSIFICATION_TYPE or self.type_of_model.lower() == model_constants.REGRESSION_TYPE:
            for model, model_name in zip(model_list, model_name_list):
                try:
                    model.fit(X_train, y_train)
                    predicted_values = model.predict(X_test)
                    if self.type_of_model.lower() == model_constants.CLASSIFICATION_TYPE:
                        metric_dict = metric_utils.get_classification_metrics(y_test, predicted_values)
                        master_metric_dict[model_name] = metric_dict
                    else:
                        metric_dict = metric_utils.get_regression_metrics(y_test, predicted_values)
                        master_metric_dict[model_name] = metric_dict
                except Exception as exception:
                    print(exception.__str__())
                    LOGGER.error('An error occurred in evaluate model %s' % exception.__str__())

        elif self.type_of_model.lower() == model_constants.CLUSTER_TYPE:
            for model_name in model_name_list:
                try:
                    cluster_labels, _ = cluster_models.get_clustered_data(X_train,clustering_algorithm=model_name)
                    metric_dict = metric_utils.get_clustering_metrics(X_train, cluster_labels)
                    master_metric_dict[model_name] = metric_dict
                except Exception as exception:
                    print(exception.__str__())
                    LOGGER.error('An error occurred in cluster part of evaluation model %s' % exception.__str__())
        return master_metric_dict

    @staticmethod
    def persist_evaluation_result(dict_to_insert):
        dbutils.check_for_duplicate_and_insert(dict_to_insert)

    def execute_pipeline(self):
        dataframe, metadata_dict = self.get_data_and_metadata()
        summary_dict = self.calculate_summary_statistics(dataframe)
        impute_algorithm_list, encoding_algorithm_list, \
        scaling_algorithm_list, transformation_algorithm_list = preprocess_utils.get_preprocessing_techiques_list()
        missing_value = metadata_dict['missing_value']
        for impute_algorithm in impute_algorithm_list:
            for encoding_algorithm in encoding_algorithm_list:
                for scaling_algorithm in scaling_algorithm_list:
                    for transformation_algorithm in transformation_algorithm_list:
                        try:
                            copy_dataframe = dataframe.copy(deep=True)
                            if str(missing_value).lower() == model_constants.NONE_MISSING_VALUES:
                                preprocessed_dataframe = self.get_processed_dataframe(copy_dataframe, metadata_dict['impute'],
                                                                                      metadata_dict['encode'],
                                                                                      metadata_dict['scale'],
                                                                                      metadata_dict['transform'],
                                                                                      impute_algorithm,
                                                                                      encoding_algorithm,
                                                                                      scaling_algorithm,
                                                                                      transformation_algorithm)
                            else:
                                preprocessed_dataframe = self.get_processed_dataframe(copy_dataframe,
                                                                                      metadata_dict['impute'],
                                                                                      metadata_dict['encode'],
                                                                                      metadata_dict['scale'],
                                                                                      metadata_dict['transform'],
                                                                                      impute_algorithm,
                                                                                      encoding_algorithm,
                                                                                      scaling_algorithm,
                                                                                      transformation_algorithm,
                                                                                      default_missing_value=missing_value)

                            model_name_list, model_list = self.get_machine_learning_models()

                            if self.type_of_model == model_constants.CLASSIFICATION_TYPE or self.type_of_model == model_constants.REGRESSION_TYPE:
                                class_labels_or_values = preprocessed_dataframe[model_constants.TARGET_COLUMN_LABEL].values
                                del preprocessed_dataframe[model_constants.TARGET_COLUMN_LABEL]
                                features = preprocessed_dataframe.values
                                X_train, X_test, y_train, y_test = split_data.get_train_test_split(features,class_labels_or_values)
                                master_metric_dict = self.evaluate_model(model_list, model_name_list, X_train, X_test, y_train, y_test)
                            else:
                                master_metric_dict = self.evaluate_model(None, model_name_list, preprocessed_dataframe.values, None, None, None)
                            for model_name in master_metric_dict.keys():
                                metric_dict = master_metric_dict[model_name]
                                merged_dict = merge_utils.merge_dicts(metric_dict, summary_dict)
                                merged_dict = merge_utils.merge_dicts(merged_dict, metadata_dict)
                                merged_dict['imputer'] = impute_algorithm if len(metadata_dict['impute']) > 0 else 'None'
                                merged_dict['encoder'] = encoding_algorithm if len(metadata_dict['encode']) > 0 else 'None'
                                merged_dict['transformer'] = transformation_algorithm if len(metadata_dict['transform']) > 0 else 'None'
                                merged_dict['scaler'] = scaling_algorithm if len(metadata_dict['scale']) > 0 else 'None'
                                merged_dict['model_name'] = model_name
                                merged_dict['model_type'] = self.type_of_model
                                merged_dict['unique_hash'] = dbutils.get_dictionary_hash(merged_dict)
                                self.persist_evaluation_result(merged_dict)
                        except Exception as exception:
                            print(exception.__str__())
                            LOGGER.error('An error occurred in execute pipeline %s ' % exception.__str__())


e = EvaluationPipeline(filepath='C:\\Users\\rupachak\\Desktop\\Papers-2019\\Large Scale Preprocessing Evaluation\\raw data\\heart.csv', type_of_model='classification')
e.execute_pipeline()
