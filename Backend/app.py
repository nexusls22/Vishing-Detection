import os.path

from DataManager import DataManager

data_folder_path = os.path.abspath(os.sep) + 'Users/Luis/Desktop/LA/LA/'

dataManager = DataManager(data_folder_path)

dataManager.check_data()
dataManager.load_data()
