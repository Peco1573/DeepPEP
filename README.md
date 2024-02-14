# DeepBEP
DeepBEP(Deep learning framework for Bacteria Essential protein Prediction) is an ESM2-based deep learning framework for bacteria essential protein prediction.
Detailed description can be found in the following paper:
Exploring the Application of  Deep Learning Model for Essential Protein Classification in Bacteria: A Comprehensive Study of  Cross-Organism Research

## Requirment
torch == 1.11.0+cu113<br>
numpy == 1.23.5<br>
pandas == 1.5.2<br>
sklearn == 1.2.1<br>


## Usage
If you want to predict the essentiality of your bacteria protein sequences, please do the following steps: <br>

Step1: ESM2 representation<br>
Download the code to your computer, paste your fasta file and pretrained ESM2 model to the ESM2_representation file, and run 'representation.py'.<br>
fasta file: containing your protein sequences<br>
pretrained ESM model: download from  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt  <br>
Note: ESM2 representation is run using CPU and requires a large amount of memory when processing longer protein sequences (it is recommended to reserve 60-100GB of memory).<br>

Step2: Ensemble test <br>
Download our pretrained models at   https://pan.baidu.com/s/1q9F5Mptz9bQbu1muttLvVQ (access code: bg03) <br>
Unziped the pretrained models to the saved_models file <br>
Run ensemble_test.py<br>
The output.csv containing sequence name, predicted essentiality and predicted score.



## Contact
