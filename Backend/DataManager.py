import numpy as np
import pandas as pd
import soundfile as sf
import os
import librosa
from pathlib import Path
from itables import init_notebook_mode, show


class DataManager:

    data_folder_path = ''

    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.train_data_path = self.data_folder_path + 'ASVspoof2019_LA_train/flac'
        self.train_protocol_path = self.data_folder_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

    def check_data(self):
        elements_in_dir = []

        print('Data Directory Contents:')
        print('------------------------')

        for i in os.listdir(self.data_folder_path):
            elements_in_dir.append(i)
            print(i)

        print('------------------------')




    def load_data(self):
        data_list = []

        if not data_list:
            print('Loading data...')
            print('------------------------')

        for i in os.listdir(self.train_data_path):
            data_list.append(i)
            print(i)

    def preprocess_data(self):
        """Preprocess data for training and evaluation."""