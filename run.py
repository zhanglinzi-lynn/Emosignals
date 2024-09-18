import collections
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import *
from utils.Trainer import *
from model.EmoSignals import *

class Run():
    def __init__(self,config):
        self.mode = config['mode']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.early_stop = config['early_stop']
        # self.device = config['device']
        self.lr = config['lr']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.path_ckp=config['path_ckp']
        self.path_tb=config['path_tb']
        self.inference_ckp=config['inference_ckp']

    def get_dataloader(self,data_path):
        dataset=Emosign_Dataset(data_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return dataloader

    def main(self):
        self.model = Emosign_Model()

        data_split_dir= 'data/data-split/'
        save_predict_result_path='./predict_result/'
            
        train_data_path=data_split_dir+'emo_train.txt'
        test_data_path=data_split_dir+'emo_test.txt'
        val_data_path=data_split_dir+'emo_val.txt'
        data_load_time_start = time.time()
        train_dataloader=self.get_dataloader(train_data_path)
        test_dataloader=self.get_dataloader(test_data_path)
        val_dataloader=self.get_dataloader(val_data_path)
        dataloaders=dict(zip(['train','test','val'],[train_dataloader,test_dataloader,val_dataloader]))
        print ('data load time: %.2f' % (time.time() - data_load_time_start))
        trainer=Trainer(model=self.model,lr=self.lr,dataloaders=dataloaders,epoches=self.epoches,model_name='Emosignals',save_predict_result_path=save_predict_result_path,beta_c=self.alpha,beta_n=self.beta,early_stop=self.early_stop,save_param_path=self.path_ckp+self.dataset+"/",writer=SummaryWriter(self.path_tb+self.dataset+"/"))
        ckp_path=trainer.train()
        result=trainer.test(ckp_path)