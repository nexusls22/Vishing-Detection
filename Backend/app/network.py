import os

import numpy as np
import torch
from torch import nn

from Backend.data_manager.DataManager import DataManager


class NeuralNetwork(nn.Module):

    def __init__(self, input_dimension):
        super().__init__()
        self.linear = nn.Linear(input_dimension, 1)

    def forward(self, x):
        return self.linear(x)

    def main(self):
        # Neural Network Model
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        train_df, dev_df = initialize_data()
        train_f64, train_f32 = initialize_features(train_df)
        dev_f64, dev_f32 = initialize_features(dev_df)

        x_tensor = torch.tensor(train_f32, dtype=torch.float32)
        y_tensor = torch.tensor(dev_f32, dtype=torch.long)

        #Map labels to 0 and 1
        label_map = {'bonafide': 0, 'spoof': 1}
        y_train = torch.tensor(train_df['label'].values, dtype=torch.float32, device=DEVICE).view(-1, 1)
        y_dev = torch.tensor(dev_df['label'].values, dtype=torch.float32, device=DEVICE).view(-1, 1)

        model = NeuralNetwork(input_dimension = x_tensor.shape[1]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(50):  # z.B. 50 Epochen
            model.train()
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


root = os.path.abspath('Backend/app')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder_path = os.path.abspath(os.sep) + 'Users/Luis/Desktop/LA/LA/'
input_sample_size = 300

#Data Managers for the specified subsets (train, dev, eval)
train_data_manager = DataManager(data_folder_path, 'train', 300)
dev_data_manager = DataManager(data_folder_path, 'dev', 300)


##For later use
#dm_eval = DataManager(data_folder_path, 'eval', 300)


#Data frames for the specified subsets (train, dev, eval)
def initialize_data(train_data_frame = None, dev_data_frame = None):

    if not train_data_frame:
        train_data_frame = train_data_manager.load_data()
    else:
        print("Train data frame already loaded")

    if not dev_data_frame:
        dev_data_frame = dev_data_manager.load_data()
    else:
        print("Dev data frame already loaded")

    return train_data_frame, dev_data_frame


##For later use
#eval_data_frame = dm_eval.load_data()

#Lists for train features data types and cast it to NumPy arrays
def initialize_features(df):

    f64 = [df['duration'],
           df['spectral_centroid_mean'].values,
           df['spectral_bandwidth_mean'].values,
           df['rolloff_mean'].values,
           df['zero_crossing_rate_mean'].values]


    f32 = [df['mean_amplitude'].values,
           df['std_amplitude'].values,
           df['max_amplitude'].values,
           df['min_amplitude'].values,
           df['mfcc_3_mean'],
           df['mfcc_3_std'],
           df['mfcc_6_mean'],
           df['mfcc_6_std'],
           df['mfcc_9_mean'].values,
           df['mfcc_9_std'].values,
           df['mfcc_12_mean'],
           df['mfcc_12_std'].values]

    return np.array(f64), np.array(f32)

