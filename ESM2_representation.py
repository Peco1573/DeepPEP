"""
input : fasta file of protein sequences,run under linux
output: *.pt of tensors stored in the "representation_results" file.
"""

import torch
import numpy as np
import pandas as pd
import esm.pretrained
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('fasta_file', type=str, help="The path to the fasta file you want to make representation")
args = parser.parse_args()
fasta_file = args.fasta_file


# fasta_file = './HCT-116.fasta'

def fasta_to_dataframe(file_name):
    f = open(file_name)
    seq = {}
    for line in f:
        if line.startswith('>'):
            name = line.replace('>', '')
            name = name.replace('\n', '')
            seq[name] = ''
        else:
            seq[name] += line.replace('\n', '')  # .strip()
    f.close()
    df = pd.DataFrame(list(seq.items()))
    df.columns = ['name', 'sequence']
    return df


df = fasta_to_dataframe(fasta_file)
if not os.path.exists('./ESM2_representation'):
    os.mkdir('./ESM2_representation')
else:
    print('Please delete ' + './ESM2_representation')
    sys.exit()
model, alphabet = esm.pretrained.load_model_and_alphabet_local('./esm2_t48_15B_UR50D.pt')
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
df['sequence'] = df['sequence'].str.slice(0, 800)
for i in range(len(df)):
    label = df['name'].iloc[i]
    data = [(label, df['sequence'].iloc[i])]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[48], return_contacts=True)
    token_representations = results['representations'][48]
    token_representation = torch.squeeze(token_representations)
    token_representation = torch.mean(token_representation, dim=0)
    name = './ESM2_representation/' + label + '.pt'
    dict1 = {label: token_representation}
    torch.save(dict1, name)
    print(i)
    print('saved:{}'.format(i))
