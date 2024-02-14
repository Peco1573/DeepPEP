import os
import torch
import pandas as pd
import data_process
import model


def test_model(test_data):
    total_y_score = []
    total_indexes = []
    with torch.no_grad():
        for i, data in enumerate(test_data, 1):
            inputs,  index = data
            inputs = inputs.float().cuda()
            output = classifier(inputs)
            output = torch.softmax(output, dim=-1)
            y_score = output[:, 1]
            total_y_score.append(y_score)
            total_indexes.append(index)
        y_score = torch.cat(total_y_score)
        y_indexes = torch.cat(total_indexes).cuda()
        df = pd.DataFrame()
        df['score'] = y_score.tolist()
        df['index'] = y_indexes.tolist()
        return df


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    # get the model list
    model_directory = "./saved_models"
    model_files_list = [f for f in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, f)) and "repeat=" in f and not os.path.splitext(f)[1]]
    # 创建test_loader
    df_representation = data_process.pt_to_dataframe()
    test_loader = data_process.test_data(df_representation)
    df_result_list = []
    for i, model_file in enumerate(model_files_list):
        classifier = model.ESM2_biLSTM_MLP().cuda()
        classifier.load_state_dict(torch.load('./saved_models/' + model_file))
        classifier.eval()
        result_df = test_model(test_loader)
        df_result_list.append(result_df)
    df_score = None
    for df in df_result_list:
        if df_score is None:
            df_score = df
        else:
            df_score = df_score.add(df, fill_value=0)
    df_score = df_score.div(len(df_result_list))
    df_output = pd.DataFrame()
    df_output['information'] = df_representation['name'].tolist()
    df_output['score'] = df_score['score'].tolist()
    df_output.to_csv('./output.csv')






