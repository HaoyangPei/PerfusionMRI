import sys
sys.path.append('/gpfs/scratch/hp2173/perfusion_MRI')

import numpy as np
import torch
import h5py

import matplotlib.pyplot as plt
import os

import importlib
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import pl_modules.GAN_module
importlib.reload(pl_modules.GAN_module)
from pl_modules.GAN_module import GAN_Module

import dataloader.BrainDCEDataset
importlib.reload(dataloader.BrainDCEDataset)
from dataloader.BrainDCEDataset import BrainDCEDataset

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out-map", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-savepath", type=str)

    opt = parser.parse_args()
    data_path = opt.data_path
    batch_size = opt.batch
    out_map = opt.out_map
    model_name = opt.model_name
    model_savepath = opt.model_savepath

    train_set = BrainDCEDataset(data_path,
                data_path+'/train.txt',
                IF_TRAIN= True)
    val_set = BrainDCEDataset(data_path,
                data_path+'/val.txt',
                IF_TRAIN= False)

    BATCH_SIZE = batch_size
    train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=BATCH_SIZE, 
                                shuffle=True,
                                num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set,
                                batch_size=BATCH_SIZE, 
                                shuffle=False,
                                num_workers=0)

    wandb_logger = WandbLogger(project="PerfusionMRI", name = model_name+'-OutMap_'+str(out_map))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAN_Module(out_map=out_map, lr=2e-4).to(device)
    checkpoint_callback1 = ModelCheckpoint(monitor='l1_loss', dirpath=model_savepath+'/l1_loss/', filename= model_name+'-OutMap_'+str(out_map)+'-{epoch:02d}-{l1_loss:.6f}')
    checkpoint_callback2 = ModelCheckpoint(monitor='g_loss', dirpath=model_savepath+'/g_loss/', filename= model_name+'-OutMap_'+str(out_map)+'-{epoch:02d}-{g_loss:.6f}')
    checkpoint_callback3 = ModelCheckpoint(dirpath=model_savepath+'/last/', filename=model_name+'-OutMap_'+str(out_map)+'-{epoch:02d}')
    trainer = pl.Trainer(gpus=1,max_epochs=200,log_every_n_steps=1,callbacks=[checkpoint_callback1,checkpoint_callback2,checkpoint_callback3],logger=wandb_logger,plugins=pl.plugins.environments.SLURMEnvironment(auto_requeue=False))
    trainer.fit(model, train_loader, val_loader)

