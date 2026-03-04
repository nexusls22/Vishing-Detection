import os.path
from itables import show

from Backend.data_manager.data_manager import data_manager

data_folder_path = os.path.abspath(os.sep) + 'Users/Luis/Desktop/LA/LA/'
input_sample_size = 300
data_manager = data_manager(data_folder_path, 'train', input_sample_size)
results = []


def main():

    get_dir_info()
    df = get_data()
    show(df, maxBytes=0)
    print(df['label'].value_counts())
    data_manager.plot_feature_distributions(df, ['zero_crossing_rate_mean', 'contrast_mean', 'rolloff_mean', 'spectral_bandwidth_mean', 'spectral_centroid_mean'])
    df.boxplot(column='duration', by='label')
    #check_data_files(df)

def get_data():
    return data_manager.load_data()

def get_dir_info():
    return data_manager.check_directory(data_folder_path)

def check_data_files(df):
    return data_manager.check_data_quality(df)