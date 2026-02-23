import os.path
from itables import init_notebook_mode
from itables_for_dash import ITable

from DataManager import DataManager

init_notebook_mode(all_interactive=True)
data_folder_path = os.path.abspath(os.sep) + 'Users/Luis/Desktop/LA/LA/'
global df

def main():
    data_manager = DataManager(data_folder_path)
    df = data_manager.load_data()

    init_notebook_mode(all_interactive=True)
    table = ITable(df, select=True)
    return table



