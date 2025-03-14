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
    return 

if __name__ == '__main__':
    # ------- configurations ------- #
    GlobalParams1=Params(param_type='k1',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=1,lr=0.0005,q=0.3)
    GlobalParams2=Params(param_type='k2',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=1,lr=0.0005, q=0.3)
    print("On: ", GlobalParams1.device)

    paser=argparse.ArgumentParser()
    paser.add_argument('--load', type=bool, default=False, help='Load the model', required=False)
    args=paser.parse_args()
    if args.load:
        load_the_model(GlobalParams1, GlobalParams2)
    
    config= Config(GlobalParams1, GlobalParams2)
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