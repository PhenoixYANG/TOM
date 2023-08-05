from src.models import *
from src import models
import pytorch_lightning as pl
import os
import logging
from src.utils import get_data
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch


def model_train(config, train_args,data):
    topic_num=data[0].get_topic_num()
    train_args['topic_num']=topic_num
    torch.set_float32_matmul_precision('high')
    if train_args['model'] == 'Vilt':
        model = MulSenti(config, train_args,data)
    elif train_args['model'] == 'Mult_2':
        model = Mult_2(config, train_args,data)
    else:
        model=getattr(models,train_args['model'])(config, train_args,data)
    if not os.path.exists(os.path.join("logs", train_args['log_dir'])):
        os.mkdir(os.path.join("logs", train_args['log_dir']))
    csv_logger = CSVLogger(save_dir='logs/', name=train_args['log_dir'])
    csv_logger.log_hyperparams(config.get_hparams())
    csv_logger.log_hyperparams(train_args)
    #'''
    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        filename='best_checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='valid_acc3',
        mode='max'
    )
    early_stop_callback = EarlyStopping(
        monitor="valid_acc3",
        mode="max",
        patience=5,
        verbose=True
    )
    '''
    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        filename='best_checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='valid_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=5,
        verbose=True
    )
    '''
    trainer = pl.Trainer(
        max_epochs=train_args['num_epochs'],
        devices=train_args['devices'],
        accelerator='gpu',
        default_root_dir='checkpoint/',
        callbacks=[early_stop_callback, 
                   model_checkpoint_callback],
        logger=csv_logger,
        strategy="ddp"
    )
    if not train_args['test']:
        trainer.fit(model)
        checkpoint_path = model_checkpoint_callback.best_model_path
    else:
        checkpoint_path=train_args['checkpoint_path']
    if train_args['model'] == 'Vilt':
        model = MulSenti.load_from_checkpoint(checkpoint_path,config=config,train_args=train_args,data=data)
    elif train_args['model'] == 'Mult_2':
        model = Mult_2.load_from_checkpoint(checkpoint_path,config=config,train_args=train_args,data=data)
    else:
        model=getattr(models,train_args['model']).load_from_checkpoint(checkpoint_path,config=config,train_args=train_args,data=data)
    trainer.test(model)
