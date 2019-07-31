from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from constants import model_constants
import pandas as pd


def get_scaled_data(dataframe, columns_to_scale='all', type_of_scaling='minmaxscaler'):
    scaler_object = get_scaler_object(type_of_scaling)
    if columns_to_scale.lower() == 'all':
        dataframe = pd.DataFrame(data=scaler_object.fit_transform(dataframe.values), columns=dataframe.columns,
                                 index=dataframe.index)
    else:
        for column_name in columns_to_scale:
            dataframe[column_name] = scaler_object.fit_transform(dataframe[column_name].values)
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


def get_encoder_object(type_of_encoding='label'):
    if type_of_encoding.lower() == model_constants.ONE_HOT_ENCODING:
        return OneHotEncoder()
    elif type_of_encoding.lower() == model_constants.LABEL_ENCODING:
        return LabelEncoder()
    else:
        return OrdinalEncoder()

# TODO - Add encoder and Imputer functions like the scaling function

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


def get_imputer_object(type_of_imputation='mean'):
    return SimpleImputer(strategy=type_of_imputation)

