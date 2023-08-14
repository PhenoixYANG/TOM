# Weakly Correlated Multimodal Sentiment Analysis: One New Dataset and One Topic-oriented Model
Code for Weakly Correlated Multimodal Sentiment Analysis: One New Dataset and One Topic-oriented Model.
![TOM](/pics/TOM.png)
## Installation

To install the necessary packages, use conda and the provided environment,yaml file:

`conda env create -f environment.yaml`

## Datasets

Our proposed RU-Senti dataset can be download from [here](https://drive.google.com/file/d/1ED1SHlYRVhduDi14-f2Xp0Mk35PdjQJU/view?usp=drive_link).

Due to concerns about data privacy, we proposed the text with [BERT tokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer) 
and extracted image features with [pretrained Vision Transformer](https://huggingface.co/google/vit-base-patch16-224).

To directly use our model without additional data processing, you can download the dataset file for [MVSA-Multiple](https://drive.google.com/drive/folders/13ZHv2d4LJa446_cZEUKMpSur7eCk3pC3?usp=drive_link) and [RU-Senti](https://drive.google.com/drive/folders/1RyU3uTA1Hbm3XxUmyvUX6q9LnSigm21T?usp=drive_link)


We expect the dataset to be placed at:

`data/RU_senti`

or

`data/MVSA_M`

## Running
You can run our code with:

`python main.py`

If you wants to use our trained checkpoint, it can be downloaded from [here]() and placed to `modelcheckpoint` dir.

To replicate expierment result on RU-Senti dataset, you should modify `model_config.use_topic=True`, `train_args['test']=True` and `train_args['dataset']=RU_senti` in the main.py file. 

To replicate expierment result on MVSA-Multiple dataset, you should modify `model_config.use_topic=True`, `train_args['test']=True` and `train_args['dataset']=MVSA_M` in the main.py file.

