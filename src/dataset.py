from turtle import shape
from transformers import ViltProcessor
import numpy as np
from torch.utils.data.dataset import Dataset
from src.TweetNormalizer import normalizeTweet
from PIL import Image
import pickle
import os
from scipy import signal
import torch
from tqdm import tqdm
import json
data_dir='data'
vilt_max_length=40
def load_photo(photo_dir):
  # global photo_features
  photo_feature_path = photo_dir + '.npy'
  photo=np.load(photo_feature_path)
  #photo_features[photo_id] = np.load(photo_feature_path)
  return photo

class Model_Dataset(Dataset):
    def __init__(self,tokenizer,data='EU_senti',split='train'):
        super(Model_Dataset,self).__init__()
        self.text=[]
        self.ids=[]
        self.image=[]
        self.label=[]
        self.senti=[0,0,0,0,0]
        self.topic=[]
        self.text_mask=[]
        self.topic_mask=[]
        data_path=os.path.join(data_dir,data,f'{data}_{split}.json')
        topics=self.load_topic(os.path.join(data_dir,data,f'{data}_topic.json'))
        self.topic_num=len(topics)
        fr=open(os.path.join(data_dir,data,f'{data}_{split}_topic.txt'))
        for line in fr:
            topic=line.split('\t')[0]
            topic=' '.join(topics[topic])
            topic=tokenizer.encode_plus(topic,add_special_tokens=True,padding='max_length',truncation=True,max_length=vilt_max_length)
            self.topic.append(topic['input_ids'])
            self.topic_mask.append(topic['attention_mask'])
        fr.close()
        fr=open(data_path,'r',encoding='utf-8')
        for i,line in tqdm(enumerate(fr)):
            review=json.loads(line)
            self.ids.append(int(review['tweet_id']))
            text=tokenizer.encode_plus(normalizeTweet(review['Text']),add_special_tokens=True,padding='max_length',truncation=True,max_length=vilt_max_length)
            self.text.append(text['input_ids'])
            self.text_mask.append(text['attention_mask'])
            self.image.append(load_photo(os.path.join(data_dir,data,'features',review['Photos'][0]['_id'])))
            self.label.append(int(review['Rating'])-1)
            self.senti[int(review['Rating'])-1]+=1
        fr.close()
        '''
        fr=open(os.path.join(data_dir,data,f'{data}_theme.txt'),'r')
        self.theme=fr.readline()
        self.theme=tokenizer.encode(normalizeTweet(self.theme))
        self.theme=torch.tensor([self.theme])
        '''
    
    def __len__(self):
        return len(self.label)

    def get_dim(self):
        return 0,0
    
    def __getitem__(self, index):
        text=torch.tensor([self.text[index]])
        topic=torch.tensor([self.topic[index]])
        text_mask=torch.tensor([self.text_mask[index]])
        topic_mask=torch.tensor([self.topic_mask[index]])
        ids=torch.tensor(self.ids[index])
        return (text,self.image[index],topic,text_mask,topic_mask),self.label[index],ids
    
    def get_senti(self):
        return self.senti

    def load_topic(self,topicpath):
        fr=open(topicpath,'r')
        topics=fr.readline()
        topics=json.loads(topics)
        return topics
    
    def get_text_lenseq(self):
        return len(self.text[0])
    
    def get_topic_num(self):
        return self.topic_num
    
