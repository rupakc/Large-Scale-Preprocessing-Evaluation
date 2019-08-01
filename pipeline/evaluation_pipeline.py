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
    