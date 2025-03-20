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
from utils import *

class Params():
    def __init__(self,param_type, target_type,trick,loss_type, delta,w=0.25,q=0, K=0.9,lr=0.001,NumTrain=500, T=3, NT1=50, NT2=100, NT3=150, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.NumTrain=NumTrain
        self.T=T
        self.NT1=NT1
        self.NT2=NT2
        self.NT3=NT3
        self.dt=T/(NT3)
        self.delta=delta
        self.w=w
        self.K=K
        self.lr=lr
        self.device=device
        self.target_type=target_type  ## "indicator", "sigmoid", "original"
        self.trick=trick  ## "logit": yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde); "clamp": dyx=zx*yx*(1+yx)*dB ;'bce': use binary cross entropy loss with logit;
        self.loss_type=loss_type ## MSELoss, BCELoss, BCEWithLogitsLoss
        self.q=q       ## inventory stoarage cost coefficient

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
        device = self.device  #torch.device('cpu')
        model=nn.Sequential(self.fc1,
                            self.relu1,
                            self.fc2,
                            self.relu2,
                            self.fc3).to(device)
        x=model(input)
        if self.scaler_type=='minmax':
            return ((x-x.amin())/(x.amax()-x.amin())).to(device)
        if self.scaler_type=='sigmoid':
            return torch.sigmoid(x).to(device)
        if self.scaler_type==None:
            return x.to(device)

class Agents():
    def __init__(self,GlobalParams):
        self.GlobalParams=GlobalParams
        self.loss=None
        self.dB=None
        self.init_x=None
        self.init_c=None
    
        self.y1_0_model, self.y12_0_model, self.y13_0_model = None, None, None
        self.y2_0_model, self.y23_0_model = None, None
        self.y3_0_model = None

        self.zy1_models, self.zy12_models, self.zy13_models = None, None, None
        self.zy2_models, self.zy23_models = None, None
        self.zy3_models = None
        

    def create(self, 
               y1_0_model, y12_0_model, y13_0_model, 
               y2_0_model, y23_0_model,
               y3_0_model,
               zy1_models, zy12_models, zy13_models,
               zy2_models, zy23_models,
               zy3_models,
               forward_loss=[],dB=None,init_x=None,init_c=None):
        
        self.y1_0_model, self.y12_0_model, self.y13_0_model = y1_0_model, y12_0_model, y13_0_model
        self.y2_0_model, self.y23_0_model = y2_0_model, y23_0_model
        self.y3_0_model = y3_0_model

        self.zy1_models, self.zy12_models, self.zy13_models = zy1_models, zy12_models, zy13_models
        self.zy2_models, self.zy23_models = zy2_models, zy23_models,
        self.zy3_models = zy3_models

        self.loss=forward_loss
        self.dB=dB.to(self.GlobalParams.device)
        self.init_x=init_x.to(self.GlobalParams.device)
        self.init_c=init_c.to(self.GlobalParams.device) 


    def create_model_dict(self,overwrite=False):
        '''
        If overwrite==True, the existing models recorded in this module will be overwritten.
        The training data of dB, init_x and init_c are included with keys='dB', 'init_x' and 'init_c' respectively.
        Forward_loss of training data is included with key='loss'.
        '''
        model_dict={'y1_0': self.y1_0_model, 'y12_0': self.y12_0_model, 'y13_0': self.y13_0_model,
                    'y2_0': self.y2_0_model, 'y23_0': self.y23_0_model,
                    'y3_0': self.y3_0_model,
                    'zy1': self.zy1_models, 'zy12': self.zy12_models, 'zy13': self.zy13_models,
                    'zy2': self.zy2_models, 'zy23': self.zy23_models,
                    'zy3': self.zy3_models,
                    'loss':self.loss,
                    'dB':self.dB,
                    'init_x':self.init_x,
                    'init_c':self.init_c,
                    'GlobalParams':self.GlobalParams,
                }
        
        if overwrite==True:
            self.model_dict=model_dict

        return model_dict

    def save_entire_models(self, path,overwrite=False,model_dict=None):
        '''
        If overwrite==True, the existing models recorded in this module will be overwritten when calling Agents().create().
        '''
        if model_dict==None:
            model_dict=self.create_model_dict(overwrite=overwrite)
        torch.save(model_dict,path)
    
    def load_entire_models(self,path,overwrite=False):  
        '''
        If overwrite==True, the existing models recorded in this module will be overwritten when calling Agents().create().
        The training data of dB, init_x and init_c are included with keys='dB','init_x' and 'init_c' respectively.
        Forward_loss of training data is included with key='loss'.
        '''
        model_dict=torch.load(path, map_location=self.GlobalParams.device, weights_only=False)
        if overwrite==True:
            self.model_dict=model_dict
            self.create(y1_0_model = model_dict['y1_0'], y12_0_model = model_dict['y12_0'], y13_0_model = model_dict['y13_0'],
                        y2_0_model = model_dict['y2_0'], y23_0_model = model_dict['y23_0'],
                        y3_0_model = model_dict['y3_0_model'],
                        zy1_models = model_dict['zy1'], zy12_models = model_dict['zy12'], zy13_models = model_dict['zy13'],
                        zy2_models = model_dict['zy2'], zy23_models = model_dict['zy23'],
                        zy3_models = model_dict['zy3'],
                        forward_loss = model_dict['loss'],
                        dB = model_dict['dB'],
                        init_x = model_dict['init_x'],
                        init_c = model_dict['init_c'],
                    )
                    
        return model_dict


class Config():
    def __init__(self, 
                 GlobalParams1, GlobalParams2,
                 MaxEpoch=500, OptimSteps=25,
                 forward_losses=[],
                 ):
        self.GlobalParams1=GlobalParams1
        self.GlobalParams2=GlobalParams2
        self.agents1=Agents(GlobalParams=self.GlobalParams1)
        self.agents2=Agents(GlobalParams=self.GlobalParams2)

        self.MaxEpoch=MaxEpoch
        self.OptimSteps=OptimSteps
        self.forward_losses=forward_losses

        self.dt=GlobalParams1.dt
        self.NT1=GlobalParams1.NT1
        self.NT2=GlobalParams1.NT2
        self.NT3=GlobalParams1.NT3
        self.NumTrain=GlobalParams1.NumTrain
        self.learning_rate=GlobalParams1.lr
        self.device=GlobalParams1.device

        self.optimizer = None
        self.scheduler = None

        
    def config_pop1(self, agents=None ,model_dict={}):   
        self.dB1=model_dict.get('dB', SampleBMIncr(GlobalParams=self.GlobalParams1))
        self.init_x1=model_dict.get('init_x', Sample_Init(GlobalParams=self.GlobalParams1))
        self.init_c1=model_dict.get('init_c', torch.zeros_like(self.init_x1))
        if agents:
            self.agents1=agents
        else:
            self.agents1.create(y1_0_model = Network(scaler_type='sigmoid'), y12_0_model = Network(scaler_type='sigmoid'), y13_0_model = Network(scaler_type='sigmoid'),
                                y2_0_model = Network(scaler_type='sigmoid'), y23_0_model = Network(scaler_type='sigmoid'),
                                y3_0_model = Network(scaler_type='sigmoid'),
                                zy1_models = [Network() for i in range(self.NT1)], zy12_models = [Network() for i in range(self.NT1)], zy13_models = [Network() for i in range(self.NT1)],
                                zy2_models = [Network() for i in range(self.NT2)], zy23_models = [Network() for i in range(self.NT2)],
                                zy3_models = [Network() for i in range(self.NT3)],
                                forward_loss=self.forward_losses,
                                dB=self.dB1,
                                init_x=self.init_x1,
                                init_c=self.init_c1,
                            )

        self.pop1_dict={'dB':self.dB1,
                        'init_x':self.init_x1 ,
                        'init_c':self.init_c1 , 
                        'GlobalParams':self.GlobalParams1, 
                        'agents':self.agents1,
                    }
        return self.pop1_dict 
    
    def config_pop2(self, agents=None, model_dict={}):
        self.dB2=model_dict.get('dB', SampleBMIncr(GlobalParams=self.GlobalParams2))
        self.init_x2=model_dict.get('init_x', Sample_Init(GlobalParams=self.GlobalParams2))
        self.init_c2=model_dict.get('init_c', torch.zeros_like(self.init_x2))
        if agents:
            self.agents2=agents
        else:
            self.agents2.create(y1_0_model = Network(scaler_type='sigmoid'), y12_0_model = Network(scaler_type='sigmoid'), y13_0_model = Network(scaler_type='sigmoid'),
                                y2_0_model = Network(scaler_type='sigmoid'), y23_0_model = Network(scaler_type='sigmoid'),
                                y3_0_model = Network(scaler_type='sigmoid'),
                                zy1_models = [Network() for i in range(self.NT1)], zy12_models = [Network() for i in range(self.NT1)], zy13_models = [Network() for i in range(self.NT1)],
                                zy2_models = [Network() for i in range(self.NT2)], zy23_models = [Network() for i in range(self.NT2)],
                                zy3_models = [Network() for i in range(self.NT3)],
                                forward_loss=self.forward_losses,
                                dB=self.dB2,
                                init_x=self.init_x2,
                                init_c=self.init_c2,
                            )

        self.pop2_dict={'dB':self.dB2,
                        'init_x':self.init_x2 ,
                        'init_c':self.init_c2 , 
                        'GlobalParams':self.GlobalParams2, 
                        'agents':self.agents2,
                    }
        return self.pop2_dict
    
    def config_NN_params(self):
        params=[]
        model_list=list(self.agents1.create_model_dict().values())+list(self.agents2.create_model_dict().values())
        for v in model_list:
            if isinstance(v, list):
                for i in v:
                    params+=list(i.parameters())
            elif isinstance(v, nn.Module):
                params+=list(v.parameters())
        
        self.optimizer = optim.Adamax(params, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        return self.optimizer, self.scheduler