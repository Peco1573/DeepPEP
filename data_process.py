import torch
import pandas as pd
import numpy as np
import re
import os
import random
from torch.utils.data import DataLoader, Dataset


def pt_to_dataframe(file_path, device):
    dict1 = torch.load(file_path, map_location=device)
    df = pd.DataFrame(data=list(dict1.items()), columns=['information', 'tensor'])
    # df = df[df['information'].str.contains('DEG1001')]

    def information_process_DEG(string1):
        pattern_DEG = r"DEG\d+"
        DEG = re.search(pattern_DEG, string1).group()
        return DEG

    def information_process_protein_accession(string1):
        pattern_protein_accession = r"DEG\d+_(.*)_"
        # pattern_protein_accession = r"^(.*?)_"
        protein_accession = re.search(pattern_protein_accession, string1).group(1)
        return protein_accession

    def information_process_essential(string1):
        essential = string1[-1]
        if essential=='e':
            return '1'
        else:
            return '0'

    def information_process_index(string1):
        pattern_index = r"(\d+)_>DEG"
        index = re.search(pattern_index, string1).group(1)
        index = int(index)
        return index

    df['DEG_id'] = df['information'].apply(lambda x: information_process_DEG(x))
    df['protein_accession'] = df['information'].apply(lambda x: information_process_protein_accession(x))
    df['essential'] = df['information'].apply(lambda x: information_process_essential(x))
    # df['index'] = df.index
    df['index'] = df['information'].apply(lambda x: information_process_index(x))
    # df['tensor'] = df['tensor'].apply(lambda x: x*10)
    # df['tensor'] = df['tensor'].apply(lambda x: (x-torch.mean(x)) / torch.std(x))
    df.sort_values(by='index', ascending=True, inplace=True)
    df.index = df['index'].tolist()

    return df


# df = pt_to_dataframe('../prokaryote_essential.pt', 'cuda:0')
# df_index = df[df['DEG_id'] =='DEG1068']
# df_index.drop(columns=['tensor'], inplace=True)
#
# df_index.to_csv('./DEG1068_index.csv')
# # # # dict1 = torch.load('./HCT_all.pt')
# print(df)


class MyDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.z = torch.tensor(z, dtype=torch.float)

    def __getitem__(self, item):
        return self.x[item, :], self.y[item], self.z[item]

    def __len__(self):
        return self.x.size(0)


def tensor_to_numpy(tensor):
    return np.array(tensor.cpu())


def transform_data(dataframe):
    """
    input: dataframe containing label(e or n) and tensor
    output: numpy_type data and target. Every sequence represented into 5120 numbers.
    :param dataframe:
    :return:
    """
    essential_dict = {"0": 0, "1": 1}
    # padding
    # data = np.zeros(shape=(len(dataframe), 5120))
    # target = np.zeros(shape=len(dataframe))
    # for i in range(len(dataframe)):
    #     for j in range(5120):
    #         data[i][j] = dataframe['tensor'].iloc[i][j]
    #     target[i] = essential_dict[dataframe['essential'][i]]
    numpy_arrays = dataframe['tensor'].apply(lambda x: tensor_to_numpy(x))
    data = np.vstack(numpy_arrays.to_list())
    target = dataframe['essential'].apply(lambda x: essential_dict[x])
    target = np.array(target)
    index = dataframe['index'].to_list()
    indexes = [int(i) for i in index]
    indexes = np.array(indexes)
    # print(len(indexes))
    return data, target, indexes


def test_data(dataframe, batch_size=32, random_seed=0):
    """
    :param dataframe: 每种细菌做到一个dataframe中
    :param batch_size:
    :param random_seed:
    :return: 由于每种细菌的essential和non_essential数量不一样，因此将non_essential数据集分成若干份，来匹配essential.将所有的正负
    平衡的数据集保存在一个列表中
    """
    random.seed(random_seed)
    test_loader_list = []
    df_e = dataframe[dataframe['essential'] == 'e']
    df_n = dataframe[~dataframe.index.isin(df_e.index)]
    df_e = df_e.reset_index(drop=True)
    df_n = df_n.reset_index(drop=True)
    indexes_n = df_n.index.tolist()
    n_splits = int(len(df_n) / len(df_e))
    groups_n = [indexes_n[i:i + len(df_e)] for i in range(0, len(indexes_n), len(df_e))]
    # 删除最后一个不满的组
    if len(groups_n[-1]) != len(df_e):
        del groups_n[-1]
    for j, split_index in enumerate(groups_n):
        sample_df_n = df_n.iloc[split_index]
        df_test = pd.concat([df_e, sample_df_n], ignore_index=True)
        df_test['test_repeat'] = j
        # print(df_test)
        X_test, y_test, protein_index = transform_data(df_test)
        test_data = MyDataset(X_test, y_test, protein_index)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        test_loader_list.append(test_loader)
    return test_loader_list

    # X_test, y_test, protein_index = transform_data(dataframe)
    # test_data = MyDataset(X_test, y_test, protein_index)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    # return test_loader


def ensemble(dataframe, random_seed=0):
    """
    :param dataframe:
    :param random_seed:
    input:dataframe from prokaryote_essential.pt
    output:分好的dataframe列表，可以遍历列表然后把每个列表的数据拿去训练神经网络.df_test_k:包含所有k折的测试集数据
    :return:
    """
    # select essential and non_essential data
    random.seed(random_seed)
    df_list = []
    df_e = dataframe[dataframe['essential'] == '1']
    df_n = dataframe[~dataframe.index.isin(df_e.index)]
    df_e = df_e.reset_index(drop=True)
    df_n = df_n.reset_index(drop=True)
    df_test_k_fold = pd.DataFrame(columns=['information', 'tensor', 'protein_accession', 'essential', 'index', 'k', 'repeat', 'train_or_test'])
    # randomly select 20% essential data and corresponding non_essential data
    indexes_e = df_e.index.tolist()
    np.random.shuffle(indexes_e)
    # 将indexes分成5份
    groups_e = [indexes_e[i:i + int(len(df_e) / 5)] for i in range(0, len(df_e), int(len(df_e) / 5))]
    if len(groups_e) == 6:
        del groups_e[-1]
    for i, e_index in enumerate(groups_e):
        df_e_test = df_e.iloc[e_index]
        df_n_test = df_n.sample(n=len(df_e_test), random_state=random_seed)
        df_e_train = df_e[~df_e.index.isin(df_e_test.index)]
        df_n_train = df_n[~df_n.index.isin(df_n_test.index)]
        df_test = pd.concat([df_n_test, df_e_test])
        df_test = df_test.reset_index(drop=True)
        df_e_train = df_e_train.reset_index(drop=True)
        df_n_train = df_n_train.reset_index(drop=True)
        df_test['k'] = i
        df_test['repeat'] = -1
        df_test['train_or_test'] = "test"
        df_test_k_fold = pd.concat([df_test_k_fold, df_test])
        # 每个test的样品抽取完了，接下来是制作train的dataframe
        indexes_n = df_n_train.index.tolist()
        np.random.shuffle(indexes_n)
        # 将indexes分成若干份。份数计算而来
        n_splits = int(len(df_n_train) / len(df_e_train))
        groups_n = [indexes_n[i:i + len(df_e_train)] for i in range(0, len(indexes_n), len(df_e_train))]
        # 删除最后一个不满的组
        if len(groups_n[-1]) != len(df_e_train):
            del groups_n[-1]
        for j, split_index in enumerate(groups_n):
            sample_df_n = df_n.iloc[split_index]
            df_train = pd.concat([sample_df_n, df_e_train], ignore_index=True)
            df_train['k'] = i
            df_train['repeat'] = j
            df_train['train_or_test'] = "train"
            df_output = pd.concat([df_test, df_train], ignore_index=True)
            df_list.append(df_output)
    # print(df_test_k_fold)
    return df_list, df_test_k_fold


def get_data(dataframe, batch_size = 32, random_seed=0):
    df_test = dataframe[dataframe['train_or_test'] == 'test']
    dataframe = dataframe[dataframe['train_or_test'] == 'train']
    df_e = dataframe[dataframe['essential'] == '1']
    df_n = dataframe[~dataframe.index.isin(df_e.index)]
    df_e = df_e.reset_index(drop=True)
    df_n = df_n.reset_index(drop=True)
    # randomly select 20% essential data and corresponding non_essential data
    df_e_validation = df_e.sample(frac=0.2, random_state=random_seed)
    df_e_train = df_e[~df_e.index.isin(df_e_validation.index)]
    df_n_validation = df_n.sample(n=len(df_e_validation), random_state=random_seed)
    df_n_train = df_n[~df_n.index.isin(df_n_validation.index)]
    df_validation = pd.concat([df_n_validation, df_e_validation])
    df_train = pd.concat([df_e_train, df_n_train])
    # initiate index
    df_validation = df_validation.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    if not df_train.empty:
        x_train, y_train, z_train = transform_data(df_train)
        train_data = MyDataset(x_train, y_train, z_train)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    else:
        train_loader = 0
    if not df_validation.empty:
        x, y, z = transform_data(df_validation)
        validation_data = MyDataset(x, y, z)
        validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)
    else:
        validation_loader = 0
    if not df_test.empty:
        x_test, y_test, z_test = transform_data(df_test)
        test_data = MyDataset(x_test, y_test, z_test)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    else:
        test_loader = 0
    return train_loader, validation_loader, test_loader



def get_data_no_validation(dataframe, batch_size = 32):
    df_train = dataframe[dataframe['train_or_test'] == 'train']
    df_test = dataframe[dataframe['train_or_test'] == 'test']
    if not df_train.empty:
        x_train, y_train, z_train = transform_data(df_train)
        train_data = MyDataset(x_train, y_train, z_train)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    else:
        train_loader = 0
    if not df_test.empty:
        x_test, y_test, z_test = transform_data(df_test)
        test_data = MyDataset(x_test, y_test, z_test)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    else:
        test_loader = 0
    return train_loader, test_loader
# df = pt_to_dataframe('./DEG1001.pt')
# test_data(df)
# df_k_0 = df_test_k_fold[df_test_k_fold['k'] == 0]
# _, test_loader = get_data(df_k_0)
# for i, data in enumerate(test_loader):
#     print(data)


