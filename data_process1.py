import torch
import pandas as pd
import numpy as np
import re
import os
import random
from torch.utils.data import DataLoader, Dataset
import sys

def pt_to_dataframe(file_path, device):
    dict1 = torch.load(file_path, map_location=device)
    df = pd.DataFrame(data=list(dict1.items()), columns=['information', 'tensor'])

    def information_process_DEG(string1):
        pattern_DEG = r"DEG\d+"
        DEG = re.search(pattern_DEG, string1).group()
        return DEG

    def information_process_protein_accession(string1):
        pattern_protein_accession = r"DEG\d+_(.*)_"
        protein_accession = re.search(pattern_protein_accession, string1).group(1)
        return protein_accession

    def information_process_essential(string1):
        essential = string1[-1]
        return essential

    def information_process_index(string1):
        pattern_index = r"(\d+)_>DEG"
        index = re.search(pattern_index, string1).group(1)
        return index

    df['DEG_id'] = df['information'].apply(lambda x: information_process_DEG(x))
    df['protein_accession'] = df['information'].apply(lambda x: information_process_protein_accession(x))
    df['essential'] = df['information'].apply(lambda x: information_process_essential(x))
    df['index'] = df['information'].apply(lambda x: information_process_index(x))
    # df['tensor'] = df['tensor'].apply(lambda x: x*10)
    # df['tensor'] = df['tensor'].apply(lambda x: (x-torch.mean(x)) / torch.std(x))

    return df



class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __getitem__(self, item):
        return self.x[item, :], self.y[item]

    def __len__(self):
        return self.x.size(0)


def tensor_to_numpy(tensor):
    return np.array(tensor.cpu().detach())


def transform_data(dataframe):
    numpy_arrays = dataframe['tensor'].apply(lambda x: tensor_to_numpy(x))
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


