import os
import warnings
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import data_process
import data_process1
import loss_function
import model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef
import shutil
import sys


def compute_evaluate_train(y_pred, labels, y_score):  # 计算精确度、准确率、召回率、f1
    y_pred = y_pred.cpu().numpy()
    labels = labels.cpu().numpy()
    y_score = y_score.cpu().numpy()
    try:
        acc = accuracy_score(labels, y_pred)
    except ValueError:
        acc = 0
    try:
        precision = precision_score(labels, y_pred)
    except ValueError:
        precision = 0
    try:
        recall = recall_score(labels, y_pred)
    except ValueError:
        recall = 0
    try:
        F1 = f1_score(labels, y_pred)
    except ValueError:
        F1 = 0
    try:
        mcc = matthews_corrcoef(labels, y_pred)
    except ValueError:
        mcc = 0
    try:
        auprc = average_precision_score(labels, y_score)
    except ValueError:
        auprc = 0
    try:
        roc = roc_auc_score(labels, y_score)
    except ValueError:
        roc = 0
    return [acc, precision, recall, F1, roc, auprc, mcc]


def compute_evaluate(y_pred, labels, y_score):  # 计算精确度、准确率、召回率、f1
    # y_pred = y_pred.cpu().numpy()
    # labels = labels.cpu().numpy()
    # y_score = y_score.cpu().numpy()
    try:
        acc = accuracy_score(labels, y_pred)
    except ValueError:
        acc = 0
    try:
        precision = precision_score(labels, y_pred)
    except ValueError:
        precision = 0
    try:
        recall = recall_score(labels, y_pred)
    except ValueError:
        recall = 0
    try:
        F1 = f1_score(labels, y_pred)
    except ValueError:
        F1 = 0
    try:
        mcc = matthews_corrcoef(labels, y_pred)
    except ValueError:
        mcc = 0
    try:
        auprc = average_precision_score(labels, y_score)
    except ValueError:
        auprc = 0
    try:
        roc = roc_auc_score(labels, y_score)
    except ValueError:
        roc = 0
    return [acc, precision, recall, F1, roc, auprc, mcc]


def test_model_ensemble(test_data):
    total_loss = 0
    total_y_pred = []
    total_y_true = []
    total_y_score = []
    total_indexes = []
    with torch.no_grad():
        for i, data in enumerate(test_data, 1):
            inputs, labels, index = data
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            # inputs, labels = inputs.float.cuda(), labels.long.cuda()
            output = classifier(inputs)
            loss2 = criterion(output, labels)
            output = torch.softmax(output, dim=-1)
            y_pred = torch.argmax(output, dim=-1)
            # print(y_pred)
            # print(labels)
            y_score = output[:, 1]
            total_y_pred.append(y_pred)
            total_y_true.append(labels)
            total_y_score.append(y_score)
            total_indexes.append(index)
            total_loss += loss2.item()
        y_pre = torch.cat(total_y_pred)
        y_true = torch.cat(total_y_true)
        y_score = torch.cat(total_y_score)
        y_indexes = torch.cat(total_indexes).to(device)
        # return total_loss, compute_evaluate(y_pre, y_true, y_score)
        df = pd.DataFrame()
        df['pre'] = y_pre.tolist()
        df['true'] = y_true.tolist()
        df['score'] = y_score.tolist()
        df['indexes'] = y_indexes.tolist()
        return df


def test_model_cross_organism(test_data):
    total_y_score = []
    total_indexes = []
    with torch.no_grad():
        for i, data in enumerate(test_data, 1):
            inputs, index = data
            inputs = inputs.float().to(device)
            output = classifier(inputs)
            output = torch.softmax(output, dim=-1)
            y_score = output[:, 1]
            total_y_score.append(y_score)
            total_indexes.append(index)
        y_score = torch.cat(total_y_score)
        y_indexes = torch.cat(total_indexes).to(device)
        df = pd.DataFrame()
        df['score'] = y_score.tolist()
        df['index'] = y_indexes.tolist()
        return df


'''---------------------cross-organism test---------------------------------------------------------------'''


def essential_to_labels(string1):
    if string1 == 'e':
        return 1
    else:
        return 0


print('begin cross_organism testing')
device = 'cuda:0'
model_directory = "./model_select/"
model_files_list = os.listdir(model_directory)
# 创建test_loader
test_path = os.listdir('./ESM2_representation/')
df_test = pd.DataFrame(columns=['information', 'tensor'])
for names in test_path:
    dict_temp = torch.load('./ESM2_representation/' + names, map_location=device)
    df_temp = pd.DataFrame(data=list(dict_temp.items()), columns=['information', 'tensor'])
    df_test = pd.concat([df_temp, df_test], ignore_index=True)
test_loader2 = data_process1.test_data(df_test)
df_result_list = []
for i, model_file in enumerate(model_files_list):
    print(i)
    classifier = model.Model5().to(device)
    '''................................................................................................'''
    classifier.load_state_dict(torch.load(model_directory + model_file, map_location=device))
    classifier.eval()
    result_df = test_model_cross_organism(test_loader2)
    df_result_list.append(result_df)
df_score = None
for df in df_result_list:
    if df_score is None:
        df_score = df
    else:
        df_score = df_score.add(df, fill_value=0)
df_score = df_score.div(len(df_result_list))
df_output = pd.DataFrame()
df_output['information'] = df_test['information']
df_output['score'] = df_score['score'].tolist()
df_output['predict'] = df_output['score'].apply(lambda x: 1 if x > 0.5 else 0)
print(df_output)
df_output.to_csv('./result.csv')
print('------------complete-Please see result.csv--------------------------------------')
