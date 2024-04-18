# DeepPEP
DeepBEP(Deep learning framework for Prokaryotitc Essential protein Prediction) is an ESM2-based deep learning framework for prokaryotic essential protein prediction.
If you want to predict essential proteins of your interested prokaryote, you should prepare the protein sequences in fasta format, and follow the guide:

## Requirment
torch == 1.11.0+cu113<br>
numpy == 1.23.5<br>
pandas == 1.5.2<br>
sklearn == 1.2.1<br>

## Usage
If you want to predict the essentiality of your bacteria protein sequences, please do the following steps using linux-64 platform: <br>

Step1: ESM2 representation of your protein sequences<br>

    git clone https://github.com/Peco1573/DeepPEP.git
    cd DeepPEP

Download ESM2 pretrained model from  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt  to './DeepPEP/ directory' <br>
Note that ESM2 pretrained model is about 30GB. The representation process is run using CPU and requires a large amount of memory (it is recommended to reserve 60-100GB of memory)<br>
Download our training set at https://pan.baidu.com/s/1_y8Cu7wDzWrnJk8RGEUJFw?pwd=1111 and unzip our dataset to './DeepPEP/ directory' <br>
run the following code('the '**your_protein.fasta**' is your own protein file'):

    cd DeepPEP
    python ESM2_representation.py your_protein.fasta
    
It will create several representations in './DeepPEP/ESM2_representation/'  <br>

Step2: Training <br>
The trained model will be saved at file folder './DeepPEP/model_select/'  <br>

    cd DeepPEP
    python train.py

Step3: Test <br>

    cd DeepPEP
    python test.py
    
It will create a 'result.csv' to './DeepPEP/'    The **result.csv** containing sequence name, predicted essentiality and predicted score. <br>

