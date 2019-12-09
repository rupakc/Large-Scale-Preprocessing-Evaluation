import pandas as pd
import json
from constants import model_constants
import os


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


def get_datafilename_and_model_type_list(raw_data_folder_name=model_constants.RAW_DATA_FOLDER_PATH):
    master_data_filepath_list = list([])
    master_type_of_model_list = list([])
    data_directory_list = os.listdir(raw_data_folder_name)
    for data_folder_name in data_directory_list:
        data_folder_path = os.path.join(raw_data_folder_name, data_folder_name)
        filename_list = os.listdir(data_folder_path)
        csv_filename = ''
        for filename in filename_list:
            if filename.find('.csv') != -1:
                csv_filename = filename
                break
        csv_full_file_path = os.path.join(data_folder_path, csv_filename)
        metadata_json_file_path = get_meta_data_filepath(csv_full_file_path)
        metadata_json_dict = get_json_content(metadata_json_file_path)
        type_of_model = metadata_json_dict['model_type']
        master_data_filepath_list.append(csv_full_file_path)
        master_type_of_model_list.append(type_of_model)
    return master_data_filepath_list, master_type_of_model_list
