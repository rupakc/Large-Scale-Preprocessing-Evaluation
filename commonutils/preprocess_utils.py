from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from constants import model_constants
import pandas as pd
import numpy as np
import six


def get_scaled_data(dataframe, columns_to_scale='all', type_of_scaling='minmaxscaler'):
    scaler_object = get_scaler_object(type_of_scaling)
    if check_string_type(columns_to_scale) and columns_to_scale.lower() == 'all':
        dataframe = pd.DataFrame(data=scaler_object.fit_transform(dataframe.values), columns=dataframe.columns,
                                 index=dataframe.index)
    else:
        for column_name in columns_to_scale:
            dataframe[column_name] = scaler_object.fit_transform(dataframe[column_name].values.reshape(-1,1))
    return dataframe, scaler_object


def get_scaler_object(type_of_scaling='minmaxscaler'):
    if type_of_scaling.lower() == model_constants.MIN_MAX_SCALER:
        return MinMaxScaler()
    elif type_of_scaling.lower() == model_constants.MAX_ABS_SCALER:
        return MaxAbsScaler()
    elif type_of_scaling.lower() == model_constants.ROBUST_SCALER:
        return RobustScaler()
    elif type_of_scaling.lower() == model_constants.STANDARD_SCALER:
        return StandardScaler()
    else:
        return Normalizer()


def get_encoded_data(dataframe, columns_to_encode, type_of_encoding='label'):
    encoder_object = get_encoder_object(type_of_encoding)
    for column in columns_to_encode:
        if type_of_encoding.lower() == model_constants.LABEL_ENCODING:
            dataframe[column] = encoder_object.fit_transform(dataframe[column].values.reshape(-1,1))
        else:
            if type_of_encoding.lower() == model_constants.ONE_HOT_ENCODING:
                encoded_value_list = encoder_object.fit_transform(dataframe[column].values.reshape(-1,1)).toarray()
            else:
                encoded_value_list = encoder_object.fit_transform(dataframe[column].values.reshape(-1, 1))
            encoded_column_name_list = get_column_name_list(encoded_value_list)
            dataframe[encoded_column_name_list] = pd.DataFrame(encoded_value_list)
            del dataframe[column]
    return dataframe, encoder_object


def get_encoder_object(type_of_encoding='label'):
    if type_of_encoding.lower() == model_constants.ONE_HOT_ENCODING:
        return OneHotEncoder(handle_unknown='ignore')
    elif type_of_encoding.lower() == model_constants.LABEL_ENCODING:
        return LabelEncoder()
    else:
        return OrdinalEncoder()


def get_transformed_data(dataframe, columns_to_transform='all', type_of_transformation='power',
                         function_transform=None, inverse_function=None, power_degree=2):
    transformer_object = get_transformer_object(type_of_transformation, function_transform, inverse_function, power_degree)
    if check_string_type(columns_to_transform) and columns_to_transform.lower() == 'all':
        dataframe = pd.DataFrame(data=transformer_object.fit_transform(dataframe.values), columns=dataframe.columns,
                                 index=dataframe.index)
    else:
        for column in columns_to_transform:
            if type_of_transformation.lower() != model_constants.POLYNOMIAL_TRANSFORMER:
                dataframe[column] = transformer_object.fit_transform(dataframe[column].values.reshape(-1, 1))
            else:
                encoded_value_list = transformer_object.fit_transform(dataframe[column].values.reshape(-1, 1))
                encoded_column_name_list = get_column_name_list(encoded_value_list)
                dataframe[encoded_column_name_list] = pd.DataFrame(encoded_value_list)
                del dataframe[column]
    return dataframe, transformer_object


def get_transformer_object(type_of_transformation='power', function_transform=None,
                           inverse_function=None, power_degree=2):
    if type_of_transformation.lower() == model_constants.POWER_TRANSFORMER:
        return PowerTransformer()
    elif type_of_transformation.lower() == model_constants.QUANTILE_TRANSFORMER:
        return QuantileTransformer(random_state=42)
    elif type_of_transformation.lower() == model_constants.FUNCTION_TRANSFORMER:
        return FunctionTransformer(func=function_transform, inverse_func=inverse_function)
    elif type_of_transformation.lower() == model_constants.POLYNOMIAL_TRANSFORMER:
        return PolynomialFeatures(degree=power_degree)


def get_imputed_data(dataframe, columns_to_impute, type_of_imputation='mean', missing_values=np.nan):
    imputer_object = get_imputer_object(type_of_imputation, missing_values=missing_values)
    for column in columns_to_impute:
        dataframe[column] = imputer_object.fit_transform(dataframe[column].values.reshape(-1,1))
    return dataframe, imputer_object


def get_imputer_object(type_of_imputation='mean', missing_values=np.nan):
    return SimpleImputer(strategy=type_of_imputation, missing_values=missing_values)


def get_preprocessing_techiques_list():

    impute_algorithm_list= [model_constants.MEAN_IMPUTATION, model_constants.MEDIAN_IMPUTATION,
                            model_constants.MOST_FREQUENT_IMPUTATION]
    encoding_algorithm_list = [model_constants.ONE_HOT_ENCODING, model_constants.LABEL_ENCODING,
                               model_constants.ORDINAL_ENCODING]
    scaling_algorithm_list = [model_constants.MIN_MAX_SCALER, model_constants.STANDARD_SCALER,
                              model_constants.MAX_ABS_SCALER, model_constants.ROBUST_SCALER,
                              model_constants.NORMALIZER]
    transformation_algorithm_list = [model_constants.POWER_TRANSFORMER, model_constants.QUANTILE_TRANSFORMER,
                                     model_constants.POLYNOMIAL_TRANSFORMER]

    return impute_algorithm_list, encoding_algorithm_list, scaling_algorithm_list, transformation_algorithm_list


def check_string_type(data_object):
    if isinstance(data_object, six.string_types):
        return True
    return False


def get_column_name_list(value_array_list):
    num_unique_values = len(value_array_list[0])
    encoded_column_list = ['transformed_column_'+ str(i) for i in range(num_unique_values)]
    return encoded_column_list