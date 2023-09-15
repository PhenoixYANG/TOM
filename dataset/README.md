# RU-Senti Dataset

 This is an introduction about our proposed RU-Senti dataset.
![RU-senti](/pics/RU-Senti.png)

## Dataset Introduction

**RU-Senti** is a multimodal sentiment analysis dataset which contains 113,588 image-text pairs collected from twitter and all the data samples are related to the Russia-Ukraine conflict and contain timestamps. RU-Senti dataset covers the period from March 2022 to December 2022, i.e., 10 months in total, and each month contains about 10,000 image-text pairs except for April and October which contains 18,675 and 14,506 image-text pairs, respectively.
![RU-senti_time](/pics/RU-Senti_date_statistic.png)
![RU-senti_distribution](/pics/RU-Senti_distribution.png)


## Download and Use

You can download the dataset from [here](https://drive.google.com/file/d/1ED1SHlYRVhduDi14-f2Xp0Mk35PdjQJU/view?usp=drive_link).

Due to concerns about data privacy, we processed the text with [BERT tokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer) 
and extracted image features with [pretrained Vision Transformer](https://huggingface.co/google/vit-base-patch16-224).

We provide a _Json_ file which includes 113,588 dicts, each dict consists of the processed text tokens named \[\'Text_tokenized\'\], the unique tweet id of the tweet named \[\'tweet_id\'\], the date when the tweet was published named \[\'date\'\] and the sentiment polarity label named \[\'Rating\'\]. 
We set __Rating=1__ to indicate that the sentiment label of the tweet is __negative__, __Rating=2__ to indicate that the sentiment label of the tweet is __neutral__, and __Rating=3__ to indicate that the sentiment label of the tweet is __positive__.

We also provide a _tar_ file, you can find the features of all the images after unpacking this file. We save the feature of each image as a separate _npy_ file named with corresponding tweet id.

We also provide the topic information extracted from text, you can download from [here](https://drive.google.com/drive/folders/13ffGh0bXdOzkfbJg8ZUNgkP9xY_1kjry?usp=drive_link). 
The _Json_ file contains differe classes of topic words while the _txt_ file contains the unique id of each tweet and its corresponding topic class.

## Run TOM on RU-Senti

To run our proposed model on the downloaded RU-Senti dataset, you need to modify the 'src/dataset.py' file. 

If you want to directly use our model without additional data processing, you can download the pytorch dataset file for [MVSA-Multiple](https://drive.google.com/drive/folders/13ZHv2d4LJa446_cZEUKMpSur7eCk3pC3?usp=drive_link) and [RU-Senti](https://drive.google.com/drive/folders/1RyU3uTA1Hbm3XxUmyvUX6q9LnSigm21T?usp=drive_link)

We expect the dataset to be placed at:

`data/RU_senti`

or

`data/MVSA_M`.




