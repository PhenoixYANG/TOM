# Weakly Correlated Multimodal Sentiment Analysis: One New Dataset and One Topic-oriented Model
Code for Weakly Correlated Multimodal Sentiment Analysis: One New Dataset and One Topic-oriented Model.
![TOM](/pics/TOM.png)
## Installation

To install the necessary packages, use conda and the provided environment,yaml file:

`conda env create -f environment.yaml`

## Datasets

Our proposed RU-Senti dataset can be download from [here](https://drive.google.com/file/d/1ED1SHlYRVhduDi14-f2Xp0Mk35PdjQJU/view?usp=drive_link).

For a detailed introduction to our proposed RU-senti dataset, please refer to `dataset/README.md`.
## Running
You can run our code with:

`python main.py`

If you wants to use our trained checkpoint, it can be downloaded from [here](https://drive.google.com/drive/folders/10Joh7Ee-0z4wAB4fIbTzbLwb45Y9YNwI?usp=drive_link) and placed to `modelcheckpoint` dir.

To replicate expierment result on RU-Senti dataset, you should modify `model_config.use_topic=True`, `train_args['test']=True` and `train_args['dataset']=RU_senti` in the main.py file. 

To replicate expierment result on MVSA-Multiple dataset, you should modify `model_config.use_topic=True`, `train_args['test']=True` and `train_args['dataset']=MVSA_M` in the main.py file.

