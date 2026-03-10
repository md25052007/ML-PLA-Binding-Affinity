# ML-PLA: Enhancing Protein-Ligand Binding Affinity Prediction with Microenvironment and Long-Range Interaction-Aware Graph Neural Networks

This repository is the official implementation of [ML-PLA]



## Requirements

To install requirements:

```setup
python=3.9
cuda=11.3
torch==1.12.1
dgl==1.0.1+cu113
dgllife==0.3.2

```

## Dataset

The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org/). The CSAR-HiQ dataset can be downloaded [here](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/drug_target_interaction/sign).

```preprocess
python preprocess.py
```
You can also use the processed data from [this link](https://zenodo.org/records/15321538). Before training the model, please put the downloaded files into the directory (./data/).

## Training

To train the model in the paper, run this command:

```train
python trainer/train.py
```



## Evaluation

To evaluate my model on test set, run:

```eval
python prediction.py
```

