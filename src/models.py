from transformers import  ViltModel,ViltConfig
from PIL import Image
import os
import pandas as pd
from src.eval_metrics import *
from src.utils import get_data,get_pretrained_model
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from modules.transformer import TransformerEncoder
from modules.mult_2 import *

class model_config():
    def __init__(
        self,
        classifier_dropout: float=0,
        hidden_size: int=768,
        n_classes: int=3,
        proj_dim :int=40,
        model: str='dandelin/vilt-b32-mlm',
        languagemodel: str='BERT',
        use_topic: bool=True,
        topic_num: int=30,
        ):
        self.classifier_dropout=classifier_dropout
        self.hidden_size=hidden_size
        self.n_classes=n_classes
        self.proj_dim=proj_dim
        self.model=model
        self.languagemodel=languagemodel
        self.use_topic=use_topic
        self.topic_num=topic_num


    def get_hparams(self):
        return {
            'classifier_dropout' : self.classifier_dropout,
            'hidden_size' : self.hidden_size,
            'n_classes' : self.n_classes,
            'proj_dim' : self.proj_dim,
            'model' : self.model,
            'languagemodel' : self.languagemodel,
            'use_topic' : self.use_topic,
            'topic_num' : self.topic_num,
        }
class TransformerEncoder_config():
    def __init__(
        self,
        embed_dim : int=768,
        num_heads : int=8,
        layers : int=5,
        attn_dropout : float=0.1,
        relu_dropout : float=0.1,
        res_dropout : float=0.1,
        embed_dropout : float=0.25,
        attn_mask : bool=True
    ):
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.layers=layers
        self.attn_dropout=attn_dropout
        self.relu_dropout=relu_dropout
        self.res_dropout=res_dropout
        self.embed_dropout=embed_dropout
        self.attn_mask=attn_mask
    def get_configs(self):
        return self.embed_dim,self.num_heads,self.layers,self.attn_dropout,self.relu_dropout,self.res_dropout,self.embed_dropout,self.attn_mask

class MulSenti(pl.LightningModule):
    def __init__(self,config,train_args,data):
        '''
        Args:
        config : store necessary parameters
        {
            classifier_dropout
            hidden_size
            n_classes
            model
        }
        '''
        super(MulSenti,self).__init__()
        self.config=config
        self.args=train_args
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.hidden_size, 3)
        )
        self.data_train=data[0]
        self.data_valid=data[1]
        self.data_test=data[2]
        self.t_seq_len=self.data_train.get_text_lenseq()
        self.i_seq_len=1
        self.language_model,_=get_pretrained_model(language=self.config.languagemodel)
        self.language_model.eval()
        self.proj_t=nn.Conv1d(in_channels=self.t_seq_len,out_channels=self.config.proj_dim,kernel_size=1)
        self.proj_i=nn.Conv1d(in_channels=self.i_seq_len,out_channels=self.config.proj_dim,kernel_size=1)
        self.TransformerConfig=TransformerEncoder_config()
        self.CMTransformer=TransformerEncoder(
            embed_dim=self.TransformerConfig.embed_dim,
            num_heads=self.TransformerConfig.num_heads,
            layers=self.TransformerConfig.layers,
            attn_dropout=self.TransformerConfig.attn_dropout,
            relu_dropout=self.TransformerConfig.relu_dropout,
            res_dropout=self.TransformerConfig.res_dropout,
            embed_dropout=self.TransformerConfig.embed_dropout,
            attn_mask=self.TransformerConfig.attn_mask)
        self.model=ViltModel.from_pretrained(self.config.model).train()
        '''
        configuration = ViltConfig()
        self.model = ViltModel(configuration)
        configuration = self.model.config
        '''
        self.softmax=nn.Softmax(dim=1)
        self.criterion=nn.CrossEntropyLoss()
        
    def forward(self, text, image, topic,text_mask,topic_mask):
        text, image, topic = self.use_pretrained(text, image, topic,text_mask,topic_mask)
        if self.config.use_topic:
            image=self.add_topic(image,topic)
        image_mask=torch.ones(image.shape[0],image.shape[1],1,device=image.device)
        vilt_output=self.model(inputs_embeds=text,image_embeds=image,pixel_mask=image_mask).pooler_output 
        output=self.classifier(vilt_output)
        return output

    def use_pretrained(self, text, image, topic,text_mask,topic_mask):
        text = text.squeeze()
        topic = topic.squeeze()
        with torch.no_grad():
            text = self.language_model(input_ids=text,attention_mask=text_mask)['last_hidden_state']
            topic = self.language_model(input_ids=topic,attention_mask=topic_mask)['last_hidden_state']
        return text, image, topic

    def add_topic(self,image,topic):
        image=self.proj_i(image) if self.i_seq_len!=self.config.proj_dim else image
        topic=self.proj_t(topic) if self.t_seq_len!=self.config.proj_dim else topic
        topic = topic.permute(1, 0, 2)
        image = image.permute(1, 0, 2)
        img_with_topic=self.CMTransformer(image,topic,topic)
        img_with_topic = img_with_topic.permute(1, 0, 2)
        return img_with_topic

    def training_step(self,batch,batch_idx):
        x,y,_id=batch #input and label
        out=self.forward(*x)
        loss=self.criterion(out,y)
        self.log("train_loss", loss,on_step=True, on_epoch=True,sync_dist=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y,_id=batch
        out=self.forward(*x)
        valid_loss=self.criterion(out,y)
        result=self.softmax(out)
        self.log("valid_loss", valid_loss,on_step=False, on_epoch=True,sync_dist=True)
        return {'result' :result,'label':y}

    def validation_epoch_end(self, validation_step_outputs):
        results=[]
        labels=[]
        for batch_outputs in validation_step_outputs:
            results.append(batch_outputs['result'])
            labels.append(batch_outputs['label'])
        all_result=torch.cat(results,dim=0)
        all_label=torch.cat(labels,dim=0)
        del results
        del labels
        if self.args['dataset']=='RU_senti':
            acc3,f1=eval_3_senti_RU(all_result,all_label)
        else:
            acc3,f1=eval_3_senti_MVSA(all_result,all_label)
        self.log("valid_acc3",acc3,sync_dist=True)
        self.log("valid_f1score",f1,sync_dist=True)

    def test_step(self,batch,batch_idx):
        x,y,_id=batch
        out=self.forward(*x)
        test_loss=self.criterion(out,y)
        result=self.softmax(out)
        self.log("test_loss", test_loss,
         on_epoch=True,sync_dist=True)
        return {'result' :result,'label':y,'id': _id}

    def test_epoch_end(self, validation_step_outputs):
        results=[]
        labels=[]
        ids=[]
        for batch_outputs in validation_step_outputs:
            results.append(batch_outputs['result'])
            labels.append(batch_outputs['label'])
            ids.append(batch_outputs['id'])
        all_result=torch.cat(results,dim=0)
        all_label=torch.cat(labels,dim=0)
        all_id=torch.cat(ids,dim=0)
        del results
        del labels
        if self.args['dataset']=='RU_senti':
            acc3,f1=eval_3_senti_RU(all_result,all_label)
        else:
            acc3,f1=eval_3_senti_MVSA(all_result,all_label)
        self.log("test_acc3",acc3,sync_dist=True)
        self.log("test_f1score",f1,sync_dist=True)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args['lr'])
        scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            "interval": "epoch",
            "monitor": "valid_loss",
            "frequency": 1
            }
    
    def train_dataloader(self):
        return DataLoader(self.data_train,batch_size=self.args['batch_size'],shuffle=False,num_workers=self.args['num_workers'])
    
    def val_dataloader(self):
        return DataLoader(self.data_valid,batch_size=self.args['batch_size'],shuffle=False,num_workers=self.args['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.data_test,batch_size=self.args['batch_size'],shuffle=False,num_workers=self.args['num_workers'])

