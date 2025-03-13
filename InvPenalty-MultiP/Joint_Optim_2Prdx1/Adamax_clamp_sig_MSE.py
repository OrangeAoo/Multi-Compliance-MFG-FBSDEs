import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import re

import argparse
import pickle
import yaml
import os, sys
import pathlib

from Model import *
from utils import *

torch.autograd.set_detect_anomaly(True)

# -------------------------------- Global parameters -------------------------------- #
global start_time, end_time
global GlobalParams1, GlobalParams2
global main_models1, main_models2, pop1_dict, pop2_dict, optimizer, scheduler



def call_globals():
    global start_time, end_time
    global GlobalParams1, GlobalParams2
    global main_models1, main_models2, pop1_dict, pop2_dict, optimizer, scheduler

    
class Config():
    def __init__(self, GlobalParams1, GlobalParams2):
        self.GlobalParams1=GlobalParams1
        self.GlobalParams2=GlobalParams2
        self.MaxEpoch=500
        self.OptimSteps=25
        self.forward_losses=[]
        self.dt=GlobalParams1.dt
        self.NT1=GlobalParams1.NT1
        self.NT2=GlobalParams1.NT2
        self.NumTrain=GlobalParams1.NumTrain

        self.forward_losses=[]

        
    def config_pop1(self):   
        self.dB1 = SampleBMIncr(GlobalParams=GlobalParams1)
        self.init_x1=GlobalParams1.init_x
        self.init_c1=torch.zeros_like(self.init_x1)
        self.main_models1
    
    def config_pop2(self):
        self.dB2 = SampleBMIncr(GlobalParams=GlobalParams2)
        self.init_x2=GlobalParams2.init_x   
        self.init_c2=torch.zeros_like(self.init_x2)
        self.main_models2

    
    GlobalParams1.lr = GlobalParams1.lr

    #Set up main models for y0 and z (z will be list of models)
    v0_model_main1 = Network(scaler_type='sigmoid')
    u0_model_main1 = Network(scaler_type='sigmoid')
    y0_model_main1 = Network(scaler_type='sigmoid')

    zv_models_main1 = [Network() for i in range(NT1)]
    zu_models_main1 = [Network() for i in range(NT1)]
    zy_models_main1 = [Network() for i in range(NT2)]
    main_models1=Main_Models(GlobalParams=GlobalParams1)
    main_models1.create(v0_model=v0_model_main1,
                        u0_model=u0_model_main1,
                        y0_model=y0_model_main1,
                        zv_models=zv_models_main1,
                        zu_models=zu_models_main1,
                        zy_models=zy_models_main1,
                        forward_loss=forward_losses,
                        dB=dB1,
                        init_x=init_x1,
                        init_c=init_c1)

    v0_model_main2 = Network(scaler_type='sigmoid')
    u0_model_main2 = Network(scaler_type='sigmoid')
    y0_model_main2 = Network(scaler_type='sigmoid')

    zv_models_main2 = [Network() for i in range(NT1)]
    zu_models_main2 = [Network() for i in range(NT1)]
    zy_models_main2 = [Network() for i in range(NT2)]
    main_models2=Main_Models(GlobalParams=GlobalParams2)
    main_models2.create(v0_model=v0_model_main2,
                        u0_model=u0_model_main2,
                        y0_model=y0_model_main2,
                        zv_models=zv_models_main2,
                        zu_models=zu_models_main2,
                        zy_models=zy_models_main2,
                        forward_loss=forward_losses,
                        dB=dB2,
                        init_x=init_x2,
                        init_c=init_c2)

    pop1_dict={'dB':dB1,
            'init_x':init_x1 ,
            'init_c':init_c1 , 
            'GlobalParams':GlobalParams1, 
            'main_models':main_models1}

    pop2_dict={'dB':dB2,
            'init_x':init_x2 ,
            'init_c':init_c2 , 
            'GlobalParams':GlobalParams2, 
            'main_models':main_models2}
    
    #Define optimization parameters
    params=[]
    params = list(main_models1.v0_model.parameters())+\
            list(main_models1.u0_model.parameters())+\
            list(main_models1.y0_model.parameters())+\
            list(main_models2.v0_model.parameters())+\
            list(main_models2.u0_model.parameters())+\
            list(main_models2.y0_model.parameters())
    for i in range(NT1):
        params += list(main_models1.zv_models[i].parameters())
        params += list(main_models1.zu_models[i].parameters())
        params += list(main_models2.zv_models[i].parameters())
        params += list(main_models2.zu_models[i].parameters())

    for i in range(NT2):
        params += list(main_models1.zy_models[i].parameters())
        params += list(main_models2.zy_models[i].parameters())
        
    #Set up optimizer and scheduler
    optimizer = optim.Adamax(params, lr=GlobalParams1.lr)
    # optimizer = optim.SGD(params, lr=GlobalParams1.lr, momentum=0.75)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)


def train_loop():
    call_globals()
    for k in range(0,MaxEpoch):
        print("Batch Number: ", k+1)
        sloss=0
        #optimize main network wrt the foward loss
        for l in range(0,OptimSteps):
            optimizer.zero_grad()
            loss = get_foward_loss(pop1_dict=pop1_dict, pop2_dict=pop2_dict)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=params,max_norm=0.7)
            optimizer.step()
            scheduler.step()
            nloss = loss.detach().numpy()
            sloss += nloss
            # print('OptimStep: '+ str(l+1))
            # print('forward_loss: ' + str(nloss))
        avgloss = sloss/OptimSteps
        print("Average Error Est: ", avgloss)
        forward_losses.append(avgloss)
    print(f"{MaxEpoch} epochs done!")

def save_the_model(dir_path, main_models1, main_models2,log_info):
    call_globals()
    path1=pathlib.Path(dir_path,'pop1.pt')
    path2=pathlib.Path(dir_path,'pop2.pt')
    main_models1.save_entire_models(path=path1)
    main_models2.save_entire_models(path=path2)

    with open(dir_path.joinpath('log_info.txt'),'w') as f:
        yaml_str = yaml.dump(log_info, indent=4)
        f.write(yaml_str)

def load_the_model(GlobalParams1, GlobalParams2):
    call_globals()
    models1=Main_Models(GlobalParams=GlobalParams1)
    models2=Main_Models(GlobalParams=GlobalParams2)
    dir_path=pathlib.Path(os.getcwd(),
                        'Results',
                        'BestModelsSaved',
                        f'{GlobalParams1.target_type}_{GlobalParams1.lr}lr_{MaxEpoch}steps_MSE_{(GlobalParams1.w)}w_{GlobalParams1.q}q') # 0.25, 0.5, 0.75

    path1=pathlib.Path(dir_path,'pop1.pt')
    path2=pathlib.Path(dir_path,'pop2.pt')
    model_dict1=models1.load_entire_models(path=path1,overwrite=True)
    model_dict2=models2.load_entire_models(path=path2,overwrite=True)
    print(f"Loaded: {dir_path.name}")
    with open(dir_path.joinpath('log_info.txt'),'r') as f:
        log_info=(f.read())
    print(f"log_info:\n{log_info.items()}")

    dB1=model_dict1['dB']
    init_x1=model_dict1['init_x']
    init_c1=model_dict1['init_c']
    pop1_dict= {'dB':dB1,
                'init_x':init_x1,
                'init_c':init_c1,
                'GlobalParams':GlobalParams1, 
                'main_models':models1}

    dB2=model_dict2['dB']
    init_x2=model_dict2['init_x']
    init_c2=model_dict2['init_c']

    pop2_dict= {'dB':dB2,
                'init_x':init_x2 ,
                'init_c':init_c2 , 
                'GlobalParams':GlobalParams2, 
                'main_models':models2}
    dt=GlobalParams1.dt
    NT1=GlobalParams1.NT1
    NT2=GlobalParams1.NT2
    NumTrain=GlobalParams1.NumTrain
    K=GlobalParams1.K
    forward_losses=models1.loss
    return 

if __name__ == '__main__':
    # ------- configurations ------- #
    GlobalParams1=Params(param_type='k1',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=1,lr=0.0005,q=0.3)
    GlobalParams2=Params(param_type='k2',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=1,lr=0.0005, q=0.1)
    config= Config(GlobalParams1, GlobalParams2)
    print("On: ", GlobalParams1.device)

    paser=argparse.ArgumentParser()
    paser.add_argument('--load', type=bool, default=False, help='Load the model', required=False)
    args=paser.parse_args()
    if args.load:
        load_the_model(GlobalParams1, GlobalParams2)
       
    # ------- train the model ------- #
    start_time = datetime.now().strftime('%B %d - %H:%M:%S')
    train_loop()
    end_time=datetime.now().strftime('%B %d - %H:%M:%S')
    print(f"Started @ {start_time}\nSaved @ {end_time}")   ## to examine whether the loss attribute is updated in the module insteance
    dir_path=pathlib.Path(os.getcwd(),
                        'Results',
                        'BestModelsSaved',
                        f'{GlobalParams1.target_type}_{GlobalParams1.lr}lr_{MaxEpoch}steps_MSE_{(GlobalParams1.w)}w_{GlobalParams1.q}q') # 0.25, 0.5, 0.75
    if not dir_path.exists():
        dir_path.mkdir()
    print(dir_path.name)

    # ------- save the model ------- #
    log_info = {'start_time':start_time, 
                'end_time':end_time,
                'file_name':dir_path.name,
                'config': { 'GlobalParams1':GlobalParams1.__dict__,
                            'GlobalParams2':GlobalParams2.__dict__,
                            'MaxEpoch':MaxEpoch,
                            'OptimSteps':OptimSteps,
                            'optimizer':optimizer.__class__,
                            'scheduler':scheduler.__class__,
                        },
            }
    if args.load == False:
        save_the_model(dir_path, main_models1, main_models2,log_info)
    # ------- plot and save ------- #
    fig_path=pathlib.Path(os.getcwd(),
                        'Results',
                        'Figs',
                        f'sigmoid_ind_{GlobalParams1.lr}lr_{MaxEpoch}steps_MSE_{(GlobalParams1.w)}w_{GlobalParams1.q}q') # 0.25, 0.5, 0.75
    if not fig_path.exists():
        fig_path.mkdir()

    plot=plot_results(pop1_dict=pop1_dict, pop2_dict=pop2_dict, loss=forward_losses, savefigs=True, to_path=fig_path)
    plot.FwdLoss(log=True)
    plot.Inventory_And_Price()
    plot.Decomposition_Inventory()
    plot.Key_Processes()
    plot.Terminal_Convergence()