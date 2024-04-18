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


def train_model(train_data):
    total_loss = 0
    for i, data in enumerate(train_data, 1):
        inputs, labels, indexes = data
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)
        output = classifier(inputs)
        loss1 = criterion(output, labels)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        total_loss += loss1.item()
    return total_loss


def validate_model(test_data):
    total_loss = 0
    total_y_pred = []
    total_y_true = []
    total_y_score = []
    with torch.no_grad():
        for i, data in enumerate(test_data, 1):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            # inputs, labels = inputs.float.cuda(), labels.long.cuda()
            output = classifier(inputs)
            loss2 = criterion(output, labels)
            output = torch.softmax(output, dim=-1)
            y_pred = torch.argmax(output, dim=-1)
            y_score = output[:, 1]
            total_y_pred.append(y_pred)
            total_y_true.append(labels)
            total_y_score.append(y_score)
            total_loss += loss2.item()
        y_pre = torch.cat(total_y_pred)
        y_true = torch.cat(total_y_true)
        y_score = torch.cat(total_y_score)
        return total_loss, compute_evaluate_train(y_pre, y_true, y_score)


def test_model_train(test_data):
    total_loss = 0
    total_y_pred = []
    total_y_true = []
    total_y_score = []
    with torch.no_grad():
        for i, data in enumerate(test_data, 1):
            inputs, labels, indexes = data
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            # inputs, labels = inputs.float.cuda(), labels.long.cuda()
            output = classifier(inputs)
            loss2 = criterion(output, labels)
            output = torch.softmax(output, dim=-1)
            y_pred = torch.argmax(output, dim=-1)
            y_score = output[:, 1]
            total_y_pred.append(y_pred)
            total_y_true.append(labels)
            total_y_score.append(y_score)
            total_loss += loss2.item()
        y_pre = torch.cat(total_y_pred)
        y_true = torch.cat(total_y_true)
        y_score = torch.cat(total_y_score)
        return total_loss, compute_evaluate_train(y_pre, y_true, y_score)


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
            inputs,  index = data
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




if __name__ == '__main__':
    device = torch.device("cuda:0")
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    """------------------------------parameters---------------------------------------------------------------------"""
    print('importing data')
    train_DEG_list = ['DEG1001', 'DEG1002', 'DEG1006', 'DEG1007', 'DEG1008', 'DEG1010', 'DEG1011', 'DEG1012', 'DEG1013', 'DEG1014',
                      'DEG1015', 'DEG1017', 'DEG1019', 'DEG1020', 'DEG1021','DEG1022',  'DEG1023', 'DEG1024', 'DEG1026', 'DEG1028',
                      'DEG1029', 'DEG1031','DEG1032', 'DEG1033', 'DEG1034', 'DEG1035', 'DEG1036', 'DEG1037', 'DEG1038', 'DEG1040',
                      'DEG1041', 'DEG1042', 'DEG1043','DEG1044', 'DEG1045', 'DEG1046', 'DEG1050', 'DEG1051', 'DEG1052', 'DEG1053',
                      'DEG1054', 'DEG1055', 'DEG1056', 'DEG1057', 'DEG1058', 'DEG1059', 'DEG1060', 'DEG1062', 'DEG1063', 'DEG1064',
                      'DEG1065', 'DEG1066', 'DEG1067', 'DEG1068']
    columns = ['acc', 'precision', 'recall', 'test_F1', 'auc', 'auprc', 'mcc', 'repeat', 'k', 'epoch', 'model']
    file_path = '../prokaryote_essential.pt'
    df_log = pd.DataFrame(columns=columns)
    loss_weight = torch.tensor(np.array([1, 1]), dtype=torch.float)
    loss_gamma = 2
    lr = 0.001
    model_dict = {
        "Model1": model.Model1,
        "Model2": model.Model2,
        "Model3": model.Model3,
        "Model4": model.Model4,
        "Model5": model.Model5,
        "Model6": model.Model6,
        "Model7": model.Model7,
    }
    model_name = 'Model5'
    epoch = 50
    model_temp_name = './model_temp'
    model_select_name = './model_select'
    if not os.path.exists(model_temp_name):
        os.mkdir(model_temp_name)
    else:
        print('Please delete ' + model_temp_name)
        sys.exit()
    if not os.path.exists(model_select_name):
        os.mkdir(model_select_name)
    else:
        print('Please delete ' + model_select_name)
        sys.exit()
    dataframe_all = data_process.pt_to_dataframe(file_path, device)
    df_DEG = dataframe_all[dataframe_all['DEG_id'].isin(train_DEG_list)]
    df_list, df_test_k_fold = data_process.ensemble(df_DEG)
    '''--------------------------------training----------------------------------------------------------------------'''
    print("begin training")
    for i, dataframe in enumerate(df_list):
        repeat = dataframe[dataframe['train_or_test'] == 'train']['repeat'].iloc[0]
        k = dataframe['k'].iloc[0]
        print('Total {} iteration，Now {} iteration'.format(len(df_list), i))
        train_loader, validation_loader = data_process.get_data_no_validation(dataframe)
        '''classifiers.........................................................................................'''
        classifier = model_dict.get(model_name)().to(device)
        '''...........................................................................................'''
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        criterion = loss_function.FocalLoss(weight=loss_weight, gamma=loss_gamma).to(device)
        train_loss_list = []
        validate_loss_list = []
        test_loss_list = []
        df_epoch = pd.DataFrame(
            columns=columns)
        for epochs in range(epoch):
            train_loss = train_model(train_loader)
            train_loss_list.append(train_loss)
            total_loss, evaluation_validation = test_model_train(validation_loader)
            df_temp = pd.DataFrame(data=[
                [evaluation_validation[0], evaluation_validation[1], evaluation_validation[2],
                 evaluation_validation[3], evaluation_validation[4], evaluation_validation[5], evaluation_validation[6],
                 repeat, k, epochs, model_name]],
                columns=columns)
            df_epoch = pd.concat([df_epoch, df_temp], ignore_index=True)
            PATH = './model_temp_{}/repeat={}_k={}_epoch={}'.format(model_name, repeat, k, epochs)
            torch.save(classifier.state_dict(), PATH)
            df_epoch.sort_values(by='test_F1', ascending=False, inplace=True)
            df_epoch = df_epoch[0:1]
        best_model_src_path = './model_temp/repeat={}_k={}_epoch={}'.format(df_epoch['repeat'].iloc[0],
                                                                                      df_epoch['k'].iloc[0],
                                                                                      df_epoch['epoch'].iloc[0])
        best_model_dst_path = './model_select/repeat={}_k={}_epoch={}'.format(df_epoch['repeat'].iloc[0],
                                                                                        df_epoch['k'].iloc[0],
                                                                                        df_epoch['epoch'].iloc[0])
        shutil.copy2(best_model_src_path, best_model_dst_path)
        shutil.rmtree('./model_temp')
        os.mkdir('./model_temp')
        df_log = pd.concat([df_epoch, df_log], ignore_index=True)
    print('end training')




