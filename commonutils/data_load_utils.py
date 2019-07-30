import pandas as pd


def get_dataframe_from_file_name(filepath):
    dataframe = pd.read_csv(filepath,error_bad_lines=False)
    return dataframe
