#  Pytorch Template for Named Entity Recognition (BERT)
## Overview
This repository contains Pytorch implement of named entitu recognition. You can define parameters in [main.py](https://github.com/AlieZVzz/Pytorch_NER_Template/blob/master/main.py), including *batch_size*, *learning_rate*, *n_epochs* etc. The parameters will save in *work_dir*. You can also use custom config file by following command.
```bash
python main.py -c [your config file address]
```
## Argument description
*work_dir* -> Directory to save args.\
*model_name* -> Pretrained model name you want to use in embedding layer\
*logdir* -> Directory to save target v.s. prediction\
*train_type* -> Different model for NER, including PLM-BiLSTM-CRF, PLM-CRF\
*patience* -> Eealystopping patience for dev dataset
## Environment
- numpy==1.23.2
- scikit_learn==1.1.2
- torch==1.12.1
- tqdm==4.64.0
- transformers==4.18.0
- yaml~=0.2.5
- pyyaml~=6.0
- scikit-learn~=1.1.2
- pandas~=1.2.5