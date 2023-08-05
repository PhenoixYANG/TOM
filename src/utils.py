import torch
import os
from src.dataset import *
import pickle
from transformers import AutoModel, AutoTokenizer,BertModel,BertTokenizerFast 
import timm
modeldir='model'
def get_data(config,train_args, split='train'):
    data_path = os.path.join(train_args['data_path'],train_args['dataset'],f"{train_args['dataset']}_{split}.dt") #data_path=data
    dataset=train_args['dataset']
    if not os.path.exists(data_path):
        tokenizer=get_tokenizer('BERT')
        print(f"  - Creating new {split} data")
        data = Model_Dataset( tokenizer,dataset, split)#dataset.py
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model

def get_pretrained_model(language=' ',vision=' '):
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    if language=='BERTweet':
        if not os.path.exists(os.path.join(modeldir,'BERTweet.pickle')):
            fw=open(os.path.join(modeldir,'BERTweet.pickle'),'wb')
            languagemodel = AutoModel.from_pretrained("vinai/bertweet-base")
            pickle.dump(languagemodel,fw)
            fw.close()
        else:
            fr=open(os.path.join(modeldir,'BERTweet.pickle'),'rb')
            languagemodel=pickle.load(fr)
            fr.close()
    elif language=='BERT':
        if not os.path.exists(os.path.join(modeldir,'BERT.pickle')):
            fw=open(os.path.join(modeldir,'BERT.pickle'),'wb')
            languagemodel = BertModel.from_pretrained("bert-base-uncased")
            pickle.dump(languagemodel,fw)
            fw.close()
        else:
            fr=open(os.path.join(modeldir,'BERT.pickle'),'rb')
            languagemodel=pickle.load(fr)
            fr.close()
    else:
        languagemodel=None
    if vision=='ViT':
        if not os.path.exists(os.path.join(modeldir,'ViT.pickle')):
            fw=open(os.path.join(modeldir,'ViT.pickle'),'wb')
            visionmodel = timm.create_model('vit_base_patch16_224', pretrained=True)
            pickle.dump(visionmodel,fw)
            fw.close()
        else:
            fr=open(os.path.join(modeldir,'ViT.pickle'),'rb')
            visionmodel=pickle.load(fr)
            fr.close()
    else:
        visionmodel=None

    return languagemodel,visionmodel

def get_tokenizer(name):
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    name_dict={
        'BERTweet':'BERTweet_tokenizer',
        'BERT': 'BERT_tokenizer'
    }
    if name=='BERTweet':
        if not os.path.exists(os.path.join(modeldir,'BERTweet_tokenizer.pickle')):
            fw=open(os.path.join(modeldir,'BERTweet_tokenizer.pickle'),'wb')
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            pickle.dump(tokenizer,fw)
            fw.close()
        else:
            fr=open(os.path.join(modeldir,'BERTweet_tokenizer.pickle'),'rb')
            tokenizer=pickle.load(fr)
            fr.close()
    if name=='BERT':
        if not os.path.exists(os.path.join(modeldir,'BERT_tokenizer.pickle')):
            fw=open(os.path.join(modeldir,'BERT_tokenizer.pickle'),'wb')
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            pickle.dump(tokenizer,fw)
            fw.close()
        else:
            fr=open(os.path.join(modeldir,'BERT_tokenizer.pickle'),'rb')
            tokenizer=pickle.load(fr)
            fr.close()
    return tokenizer
