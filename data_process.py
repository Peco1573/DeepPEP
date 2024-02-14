import torch
import pandas as pd
import numpy as np
import re
import os
import random
from torch.utils.data import DataLoader, Dataset
import sys


def pt_to_dataframe():
    df_output = pd.DataFrame(columns=['name', 'representation'])
    list_dir = os.listdir('./ESM2_representation/representation_results')
    list_pt = [file for file in list_dir if file.endswith('.pt')]
    if len(list_pt) == 0:
        print('no representation results')
        sys.exit()
    else:
        for i in range(len(list_pt)):
            dict_temp = torch.load('./ESM2_representation/representation_results/' + list_pt[i])
            df_temp = pd.DataFrame(data=dict_temp.items(), columns=['name', 'representation'])
            df_output = pd.concat([df_temp, df_output], ignore_index=True)
    return df_output


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __getitem__(self, item):
        return self.x[item, :], self.y[item]

    def __len__(self):
        return self.x.size(0)


def tensor_to_numpy(tensor):
    return np.array(tensor)


def transform_data(dataframe):
    numpy_arrays = dataframe['representation'].apply(lambda x: tensor_to_numpy(x))
    data = np.vstack(numpy_arrays.to_list())
    index = dataframe.index.to_list()
    indexes = [int(i) for i in index]
    indexes = np.array(indexes)
    return data, indexes


def test_data(dataframe, batch_size=32, random_seed=0):
    data, indexes = transform_data(dataframe)
    test_data = MyDataset(data, indexes)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return test_loader


