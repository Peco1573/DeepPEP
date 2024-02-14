import torch
import numpy as np
import pandas as pd
import esm.pretrained
import os
import sys


def parse_fasta_to_df(fasta_file):
    sequences = {'name': [], 'sequence': []}
    current_name = ""
    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                current_name = line[1:]
                continue
            sequences['name'].append(current_name)
            sequences['sequence'].append(line)

    df = pd.DataFrame(sequences)
    file_name = os.path.splitext(fasta_file)[0]
    df['file_name'] = file_name  # 添加文件名列
    return df


def make_dataframe():
    file_path = './'
    dataframe = pd.DataFrame(columns=['name', 'sequence', 'file_name'])
    list_fasta = []
    for filename in os.listdir(file_path):
        if filename.endswith('.fasta'):
            list_fasta.append(filename)
    if len(list_fasta) == 0:
        print('no fasta_file_found')
        sys.exit()
    else:
        print(list_fasta)
    for i, name in enumerate(list_fasta):
        df_temp = pd.DataFrame(columns=['name', 'sequence', 'file_name'])
        df_temp = parse_fasta_to_df(name)
        dataframe = pd.concat([dataframe, df_temp], ignore_index=True)
    return dataframe


model, alphabet = esm.pretrained.load_model_and_alphabet_local('./esm2_t48_15B_UR50D.pt')
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
df = make_dataframe()
df['sequence'] = df['sequence'].str.slice(0, 800)
for i in range(len(df)):
    label = df['file_name'].iloc[i] + '_' + df['name'].iloc[i]
    data = [(label, df['sequence'].iloc[i])]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[48], return_contacts=True)
    token_representations = results['representations'][48]
    token_representation = torch.squeeze(token_representations)
    token_representation = torch.mean(token_representation, dim=0)
    name = './representation_results/' + label + '.pt'
    dict1 = {label: token_representation}
    torch.save(dict1, name)
    print('saved:{}'.format(i))
