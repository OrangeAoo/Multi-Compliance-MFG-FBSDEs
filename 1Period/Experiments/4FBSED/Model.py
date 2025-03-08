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
    def __init__(self,param_type, target_type,trick, delta,K,lr=0.005,NumTrain=500, NT=80, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.NumTrain=NumTrain
        self.NT=NT
        self.dt=1/NT
        self.delta=delta
        self.K=K
        self.lr=lr
        self.device=device
        self.target_type=target_type  ## "indicator", "sigmoid", "original"
        self.trick=trick  ## "logit": yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde); "clamp": dyx=zx*yx*(1+yx)*dB ;'bce': use binary cross entropy loss with logit;
        # self.Loss_type=Loss_type ## nn.MSELoss, nn.BCEWithLogitsLoss

        if param_type=='k1':
            #k1
            self.h=0.2
            self.zeta=1.75
            self.beta=1
            self.gamma=1.25
            self.sigma=0.1
            self.mean=0.6
            self.std=0.1

        if param_type=='k2':
            #k2
            self.h=0.5
            self.zeta=1.25
            self.beta=1
            self.gamma=1.75
            self.sigma=0.15
            self.mean=0.2
            self.std=0.1
    

class Network(nn.Module):
    def __init__(self, lr=0.005, input_dims=[1], fc1_dims=10, fc2_dims=10, n_outputs=1):
        '''
        lr: learning rate
        '''
        super(Network, self).__init__()

        #Pass input parameters
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_out = n_outputs

        #Construct network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_out)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.model=nn.Sequential(self.fc1,
                                 nn.ReLU(),
                                 self.fc2,
                                 nn.ReLU(),
                                 self.fc3)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)

    # def forward(self, input):
    #     x = F.relu(self.fc1(input))
    #     x= F.relu(self.fc2(x))
    #     output = self.fc3(x)
    #     return output.to(self.device)
    def forward(self,input):
        return self.model(input).to(self.device)

class Main_Models():
    def __init__(self,GlobalParams):
        self.GlobalParams=GlobalParams
        self.yx0_model=None
        self.yc0_model=None
        self.zx_models=None
        self.zc_models=None
    
    def create(self, yx0_model,yc0_model,zx_models,zc_models,forward_loss=None,dB=None,init_x=None,init_c=None):
        self.yx0_model=yx0_model
        self.yc0_model=yc0_model
        self.zx_models=zx_models
        self.zc_models=zc_models
        self.loss=forward_loss
        self.dB=dB
        self.init_x=init_x
        self.init_c=init_c


    def create_model_dict(self,overwrite=False):
        '''
        If overwrite==True, the existing models recorded in this module will be overwritten.
        The training data of dB, init_x and init_c are included with keys='dB', 'init_x' and 'init_c' respectively.
        Forward_loss of training data is included with key='loss'.
        '''
        model_dict={"yx0":self.yx0_model,
                    'yc0':self.yc0_model,
                    'zxs':self.zx_models,
                    'zcs':self.zc_models,
                    'loss':self.loss,
                    'dB':self.dB,
                    'init_x':self.init_x,
                    'init_c':self.init_c,
                    'GlobalParams':self.GlobalParams}
        # for i in range(len(self.zx_models)):
        #     model_dict[f'zx{i}']=self.zx_models[i]
        #     model_dict[f'zc{i}']=self.zc_models[i]
        
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
        model_dict=torch.load(path)
        # zx_models=[model_dict[f'zx{i}'] for i in range(len((model_dict-2)/2))]
        # zc_models=[model_dict[f'zc{i}'] for i in range(len((model_dict-2)/2))]
        if overwrite==True:
            self.model_dict=model_dict
            self.create(model_dict['yx0'],model_dict['yc0'],model_dict['zxs'],model_dict['zcs'])
            
        return model_dict
