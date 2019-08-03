import pandas as pd


def merge_dicts(dict_one, dict_two):
    merged_dict = {**dict_one, **dict_two}
    return merged_dict


def get_complete_dataframe_from_dict(dict_list):
    dataframe_list = list([])
    for dictionary_object in dict_list:
        dataframe_list.append(pd.DataFrame(dictionary_object))
    return pd.concat(dataframe_list, axis=0, ignore_index=False)
