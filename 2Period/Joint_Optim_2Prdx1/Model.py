import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from scipy.stats import norm

class Params():
    def __init__(self,param_type, target_type,trick,loss_type, delta,w=1.0,K=0.9,lr=0.005,NumTrain=500, T=2, NT1=50, NT2=100, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.NumTrain=NumTrain
        self.T=T
        self.NT1=NT1
        self.NT2=NT2
        self.dt=T/(NT2)
        self.delta=delta
        self.w=w
        self.K=K
        self.lr=lr
        self.device=device
        self.target_type=target_type  ## "indicator", "sigmoid", "original"
        self.trick=trick  ## "logit": yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde); "clamp": dyx=zx*yx*(1+yx)*dB ;'bce': use binary cross entropy loss with logit;
        self.loss_type=loss_type ## MSELoss, BCELoss, BCEWithLogitsLoss
        
        if param_type=='k1':
            #k1
            self.pi=0.25
            self.h=0.2
            self.zeta=1.75
            self.beta=1
            self.gamma=1.25
            self.sigma=0.1
            self.mean=0.6
            self.std=0.1
            

        if param_type=='k2':
            #k2
            self.pi=0.75
            self.h=0.5
            self.zeta=1.25
            self.beta=1
            self.gamma=1.75
            self.sigma=0.15
            self.mean=0.2
            self.std=0.1
    

class Network(nn.Module):
    def __init__(self, scaler_type=None, input_dims=1, fc1_dims=10, fc2_dims=10, n_outputs=1):
        super(Network, self).__init__()

        #Pass input parameters
        self.scaler_type=scaler_type
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_out = n_outputs

        #Construct network
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.relu1=nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(self.fc2_dims, self.n_out)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self,input):
        model=nn.Sequential(self.fc1,
                            self.relu1,
                            self.fc2,
                            self.relu2,
                            self.fc3).to(self.device)
        x=model(input)
        if self.scaler_type=='minmax':
            return ((x-x.amin())/(x.amax()-x.amin())).to(self.device)
        if self.scaler_type=='sigmoid':
            return torch.sigmoid(x).to(self.device)
        if self.scaler_type==None:
            return x.to(self.device)

class Main_Models():
    def __init__(self,GlobalParams):
        self.GlobalParams=GlobalParams
        self.loss=None
        self.dB=None
        self.init_x=None
        self.init_c=None
    
        self.v0_model=None
        self.u0_model=None
        self.y0_model=None
        self.zv_models=None
        self.zu_models=None
        self.zy_models=None

    def create(self, v0_model,u0_model,y0_model,zv_models,zu_models,zy_models,forward_loss=None,dB=None,init_x=None,init_c=None):
        self.loss=forward_loss
        self.dB=dB
        self.init_x=init_x
        self.init_c=init_c
        
        self.v0_model=v0_model
        self.u0_model=u0_model
        self.y0_model=y0_model
        self.zv_models=zv_models
        self.zu_models=zu_models
        self.zy_models=zy_models
  

    def create_model_dict(self,overwrite=False):
        '''
        If overwrite==True, the existing models recorded in this module will be overwritten.
        The training data of dB, init_x and init_c are included with keys='dB', 'init_x' and 'init_c' respectively.
        Forward_loss of training data is included with key='loss'.
        '''
        model_dict={'v0': self.v0_model,
                    'u0': self.u0_model,
                    'y0': self.y0_model,
                    'zvs': self.zv_models,
                    'zus': self.zu_models,
                    'zys': self.zy_models,
                    'loss':self.loss,
                    'dB':self.dB,
                    'init_x':self.init_x,
                    'init_c':self.init_c,
                    'GlobalParams':self.GlobalParams}
        
        if overwrite==True:
            self.model_dict=model_dict
        return model_dict

    def save_entire_models(self, path,overwrite=False,model_dict=None):
        '''
        If overwrite==True, the existing models recorded in this module will be overwritten when calling Main_Models().create().
        '''
        if model_dict==None:
            model_dict=self.create_model_dict(overwrite=overwrite)
        torch.save(model_dict,path)
    
    def load_entire_models(self,path,overwrite=False):  
        '''
        If overwrite==True, the existing models recorded in this module will be overwritten when calling Main_Models().create().
        The training data of dB, init_x and init_c are included with keys='dB','init_x' and 'init_c' respectively.
        Forward_loss of training data is included with key='loss'.
        '''
        model_dict=torch.load(path,weights_only=True)
        if overwrite==True:
            self.model_dict=model_dict
            self.create(v0_model=model_dict['v0'],
                        u0_model=model_dict['u0'],
                        y0_model=model_dict['y0'],
                        zv_models=model_dict['zvs'],
                        zu_models=model_dict['zus'],
                        zy_models=model_dict['zys'],
                        forward_loss=model_dict['loss'],
                        dB=model_dict['dB'],
                        init_x=model_dict['init_x'],
                        init_c=model_dict['init_c'])
            
        return model_dict
