# DeepPEP
DeepBEP(Deep learning framework for Prokaryotitc Essential protein Prediction) is an ESM2-based deep learning framework for prokaryotic essential protein prediction.
If you want to predict essential proteins of your interested prokaryote, you should prepare the protein sequences in fasta format, and follow the guide:

## Requirment
torch == 1.11.0+cu113<br>
numpy == 1.23.5<br>
pandas == 1.5.2<br>
sklearn == 1.2.1<br>

## Usage
If you want to predict the essentiality of your bacteria protein sequences, please do the following steps: <br>

Step1: ESM2 representation of your protein sequences<br>
Download our code to your computer, paste your fasta file to ./DeepPEP/ directory<br>
Download ESM2 pretrained model from  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt  to ./DeepPEP/ directory <br>
Note that ESM2 pretrained model is about 30GB. The representation process is run using CPU and requires a large amount of memory (it is recommended to reserve 60-100GB of memory)<br>
run the following ESM2_

## Usage

If you want to predict the essentiality of your bacteria protein sequences, please do the following steps using linux-64 platform.
#### 1. Clone the repo


    git clone https://github.com/lynn-1998/DeepCellEss.git
    cd DeepCellEss


#### 2. Create and activate the environment

    cd DeepCellEss
    conda create --name deepcelless --file requirments.txt
    conda activate deepcelless


#### 3. Train model
The trained models will be saved at file folder '../protein/saved_model/HCT-116/'.

    cd code
    python main.py protein --cell_line HCT-116 --gpu 0


Step2: Ensemble test <br>
Download our pretrained models at   https://pan.baidu.com/s/1q9F5Mptz9bQbu1muttLvVQ (access code: bg03) <br>
Unziped the pretrained models to the saved_models file <br>
Run ensemble_test.py<br>
The output.csv containing sequence name, predicted essentiality and predicted score.



## Contact
