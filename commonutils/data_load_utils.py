import pandas as pd
import json
from constants import model_constants


def get_dataframe_from_file_name(filepath):
    dataframe = pd.read_csv(filepath,error_bad_lines=False)
    return dataframe


def get_json_content(filepath):
    json_content = {}
    with open(filepath, 'r', encoding='utf-8') as json_file:
        json_content = json.load(json_file)
    return json_content


def get_meta_data_filepath(filepath):
    modified_filename = filepath.replace(model_constants.DATA_FILE_EXTENSION, '')
    modified_filename = modified_filename + model_constants.META_DATA_FILE_SUFFIX + model_constants.META_DATA_FILE_EXTENSION
    return modified_filename
