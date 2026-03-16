import os.path
from itables import show

from Backend.data.datamanager import DataManager, check_directory, check_data_quality

data_folder_path = os.path.abspath(os.sep) + 'Users/Luis/Desktop/LA/LA/'
input_sample_size = 300
data_manager = DataManager(data_folder_path, 'train', input_sample_size)
results = []

## For next step: Proper classes for regression, learning etc
#Backend/
#├── data/
#│   └── datamanager.py     # DataManager (Daten laden/validieren)
#├── network/
#│   ├── __init__.py
#│   ├── models.py          # Neural Network Definition
#│   ├── regression.py      # Sklearn-Models
#│   ├── trainer.py         # Train logic with epochs
#│   └── utils.py           # Helper functions
#└── app.py                 # Streamlit UI (only Presentation)

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
    return check_directory(data_folder_path)

def check_data_files(df):
    return check_data_quality(df)