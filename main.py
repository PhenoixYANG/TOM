from src.models import model_config
from src.dataset import Model_Dataset
from src.train import model_train
from torch.utils.data import DataLoader
from src.utils import get_data
import argparse
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def get_logdir(args):
    if args['model']=='Vilt':
        if args['dataset']=='EU-senti':
            args['log_dir']='MulSentiVilt'
        elif args['dataset']=='MVSA_M':
            args['log_dir']='MulSentiViltMVSA'
        elif args['dataset']=='RU_senti':
            args['log_dir']='RU_SentiVilt'
    elif args['model']=='Mult_2': 
        if args['dataset']=='EU-senti':
            args['log_dir']='MulSentiMult2'
        elif args['dataset']=='MVSA_M':
            args['log_dir']='MulSentiMult2_MVSA'
        elif args['dataset']=='RU_senti':
            args['log_dir']='RU_SentiMult2'
    else:
        args['log_dir']=f'{args["model"]}_{args["dataset"]}'
    return args
if __name__=='__main__':      
    config=model_config(
        classifier_dropout=0.1,
        hidden_size=768,
        n_classes=3,
        proj_dim=40,
        model='dandelin/vilt-b32-mlm',
        languagemodel= 'BERT',
        use_topic=False,
        #use_topic=True,
        topic_num=30,
        )   
    train_args={
        'test' : True, #use checkpoint to test or not
        'dataset' : 'RU_senti',
        'data_path' : 'data',
        'num_epochs' : 60,
        'batch_size' : 128,
        'num_workers' :64,
        'lr' : 0.0005,
        'devices' :[8],
        'log_dir' :'MulSentiViltMVSA',
        'model' :'Vilt',
        'checkpoint_path' : os.path.join('modelcheckpoint','ToMSAM.ckpt'),
        'loops' :1
    }
    train_args=get_logdir(train_args)
    data_train=get_data(config,train_args,'train')
    data_valid=get_data(config,train_args,'valid')
    data_test=get_data(config,train_args,'test')
    data=[data_train,data_valid,data_test]
    for i in range(0,train_args['loops']):
        print(f'loop:{i+1}')
        model_train(config,train_args,data)


