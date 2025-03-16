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
global agents1, agents2, pop1_dict, pop2_dict, optimizer, scheduler
global MaxEpoch, OptimSteps
# sys.path.append(pathlib.Path("/Users/orangeao/OrangeAo/ResearchAndProjects/Multi-Compliance-MFG-FBSDEs/InvPenalty-MultiP/Joint_Optim_2Prdx1"))

os.chdir("./InvPenalty-MultiP/Joint_Optim_2Prdx1")
print("Current working dir: ", os.getcwd())
# ----------------------------------------------------------------------------------- #


def train_loop(configs):
    MaxEpoch = configs.MaxEpoch
    OptimSteps = configs.OptimSteps
    optimizer = configs.optimizer
    scheduler = configs.scheduler
    forward_losses = []
    for k in range(0,MaxEpoch):
        print("Batch Number: ", k+1)
        sloss=0
        #optimize main network wrt the forward loss
        for l in range(0,OptimSteps):
            optimizer.zero_grad()
            loss = get_forward_loss(pop1_dict=configs.pop1_dict, pop2_dict=configs.pop2_dict)
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

    configs.forward_losses = forward_losses
    return configs, configs.agents1, configs.agents2, forward_losses

def save_the_model(dir_path, configs,log_info):
    path1=pathlib.Path(dir_path,'pop1.pt')
    path2=pathlib.Path(dir_path,'pop2.pt')
    configs.agents1.save_entire_models(path=path1)
    configs.agents2.save_entire_models(path=path2)

    with open(dir_path.joinpath('log_info.txt'),'w') as f:
        yaml_str = yaml.dump(log_info, indent=4)
        f.write(yaml_str)
    return 


def load_the_model(dir_path, configs):
    GlobalParams1 = configs.GlobalParams1
    GlobalParams2 = configs.GlobalParams2

    agents1=Agents(GlobalParams=GlobalParams1)
    agents2=Agents(GlobalParams=GlobalParams2)
    dir_path=pathlib.Path(os.getcwd(),
                        'Results',
                        'BestModelsSaved',
                        f'{GlobalParams1.target_type}_{GlobalParams1.lr}lr_{configs.MaxEpoch}steps_MSE_{(GlobalParams1.w)}w_{GlobalParams1.q}q') # 0.25, 0.5, 0.75

    path1=pathlib.Path(dir_path,'pop1.pt')
    path2=pathlib.Path(dir_path,'pop2.pt')
    model_dict1=agents1.load_entire_models(path=path1,overwrite=True)
    model_dict2=agents2.load_entire_models(path=path2,overwrite=True)
    print(f"Loaded: {dir_path.name}")

    with open(dir_path.joinpath('log_info.txt'),'r') as f:
        log_info=yaml.load(f.read(),Loader=yaml.FullLoader)
        for i, (k,v) in enumerate(log_info.items()):
            if i>=3: break
            print(f"{k}:\t{v}")
    return {"agents1":agents1, 
            "agents2":agents2,
            "model_dict1":model_dict1,
            "model_dict2":model_dict2,}

if __name__ == '__main__':
    # ------- configurations ------- #
    GlobalParams1=Params(param_type='k1',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=1,lr=0.0005,q=0.1)
    GlobalParams2=Params(param_type='k2',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=1,lr=0.0005, q=0.1)
    print("On: ", GlobalParams1.device)

    parser=argparse.ArgumentParser()
    parser.add_argument('--load', type=bool, default=False, help='Load the model', required=False)
    args=parser.parse_args()

    configs= Config(GlobalParams1, GlobalParams2)
    dir_path=pathlib.Path(os.getcwd(),
                            'Results',
                            'BestModelsSaved',
                            f'{GlobalParams1.target_type}_{GlobalParams1.trick}_{GlobalParams1.lr}lr_{configs.MaxEpoch}steps_MSE_{(GlobalParams1.w)}w_{GlobalParams1.q}q') # 0.25, 0.5, 0.75
        
    if args.load ==False:
        configs.config_pop1() 
        configs.config_pop2()
        configs.config_NN_params()
        # ------- train the model ------- #
        start_time = datetime.now().strftime('%B %d - %H:%M:%S')
        configs.forward_losses = train_loop(configs)
        end_time=datetime.now().strftime('%B %d - %H:%M:%S')
        print(f"Started @ {start_time}\nSaved @ {end_time}")   ## to examine whether the loss attribute is updated in the module insteance

        # ------- save the model ------- #
        if not dir_path.exists():
            dir_path.mkdir()
        print(dir_path.name)

        log_info = {'start_time':start_time, 
                    'end_time':end_time,
                    'file_name':dir_path.name,
                    'config': { 'GlobalParams1':GlobalParams1.__dict__,
                                'GlobalParams2':GlobalParams2.__dict__,
                                'MaxEpoch':configs.MaxEpoch,
                                'OptimSteps':configs.OptimSteps,
                                'optimizer':configs.optimizer.__class__,
                                'scheduler':configs.scheduler.__class__,
                            },
                }
        save_the_model(dir_path, configs, log_info)
        fig_path=pathlib.Path(os.getcwd(),
                        'Results',
                        'Figs',
                        f'{GlobalParams1.target_type}_{GlobalParams1.trick}_{GlobalParams1.lr}lr_{configs.MaxEpoch}steps_MSE_{(GlobalParams1.w)}w_{GlobalParams1.q}q') # 0.25, 0.5, 0.75
    
    else:
        loaded_res=load_the_model(dir_path, configs)
        configs.config_pop1(loaded_res['agents1'], loaded_res['model_dict1'])
        configs.config_pop2(loaded_res['agents2'], loaded_res['model_dict2'])
        configs.forward_losses = loaded_res['agents1'].loss
        fig_path=pathlib.Path(os.getcwd(),
                        'Results',
                        'Figs',
                        f'Loaded_{GlobalParams1.target_type}_{GlobalParams1.trick}_{GlobalParams1.lr}lr_{configs.MaxEpoch}steps_MSE_{(GlobalParams1.w)}w_{GlobalParams1.q}q') # 0.25, 0.5, 0.75
    

    # ------- plot and save ------- #
    if not fig_path.exists():
        fig_path.mkdir()

    plot=plot_results(pop1_dict=configs.pop1_dict, 
                      pop2_dict=configs.pop2_dict, 
                      loss=configs.forward_losses, 
                      savefigs=True, to_path=fig_path)
    plot.FwdLoss(log=True)
    plot.Inventory_And_Price()
    plot.Decomposition_Inventory()
    plot.Key_Processes()
    plot.Terminal_Convergence()