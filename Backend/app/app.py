import os.path
from itables import init_notebook_mode, show

from Backend.data_manager.DataManager import DataManager

init_notebook_mode(all_interactive=True)
data_folder_path = os.path.abspath(os.sep) + 'Users/Luis/Desktop/LA/LA/'
results = []

def main():
    data_manager = DataManager(data_folder_path)
    get_dir_info(data_manager)
    #get_files_list(data_manager)


def get_i_table():
    data_manager = DataManager(data_folder_path)
    df, f_df = data_manager.load_data(results, data_manager.input_sample_size)

    return df, f_df

def get_dir_info(data_manager):
    return data_manager.check_data_dir()

def get_files_list(data_manager):
    return data_manager.check_data_files()