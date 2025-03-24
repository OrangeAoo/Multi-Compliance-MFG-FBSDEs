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
from pathlib import Path

from Model import *
from utils import *

torch.autograd.set_detect_anomaly(True)

# -------------------------------- Global parameters -------------------------------- #
global start_time, end_time
global GlobalParams1, GlobalParams2
global agents1, agents2, pop1_dict, pop2_dict, optimizer, scheduler
global MaxEpoch, OptimSteps
# sys.path.append(Path("/Users/orangeao/OrangeAo/ResearchAndProjects/Multi-Compliance-MFG-FBSDEs/InvPenalty-MultiP/Joint_Optim_2Prdx1"))

# os.chdir("./InvPenalty-MultiP/Joint_Optim_2Prdx1")
print("Current working dir: ", os.getcwd())
# ----------------------------------------------------------------------------------- #


def train_loop(configs,agents1, agents2):
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
            loss = get_forward_loss(agents1, agents2)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=params,max_norm=0.7)
            optimizer.step()
            scheduler.step()
            nloss = loss
            sloss += nloss
            # print('OptimStep: '+ str(l+1))
            # print('forward_loss: ' + str(nloss))
        avgloss = (sloss/OptimSteps).cpu().detach().numpy()
        print("Average Error Est: ", avgloss)
        forward_losses.append(avgloss)
    print(f"{MaxEpoch} epochs done!")

    agents1.loss, agents2.loss = forward_losses, forward_losses
    return forward_losses, agents1, agents2, configs

def save_the_model(dir_path, agents1, agents2, log_info):
    path1=Path(dir_path,'pop1.pt')
    path2=Path(dir_path,'pop2.pt')
    agents1.save_entire_models(path=path1)
    agents2.save_entire_models(path=path2)

    with open(dir_path.joinpath('log_info.txt'),'w') as f:
        yaml_str = yaml.dump(log_info, indent=4)
        f.write(yaml_str)
    return 


def load_the_model(dir_path, configs):
    dir_path=Path(dir_path)
    GlobalParams1 = configs.GlobalParams1
    GlobalParams2 = configs.GlobalParams2

    agents1=Agents(GlobalParams=GlobalParams1)
    agents2=Agents(GlobalParams=GlobalParams2)

    model_dict1=agents1.load_entire_models(path=dir_path.joinpath('pop1.pt'),overwrite=True)
    model_dict2=agents2.load_entire_models(path=dir_path.joinpath('pop2.pt'),overwrite=True)
    agents1.reset_configs(model_dict=model_dict1)
    agents2.reset_configs(model_dict=model_dict2)

    with open(dir_path.joinpath('log_info.txt'),'r') as f:
        log_info=yaml.load(f.read(),Loader=yaml.FullLoader)
        assert (log_info['file_name'] == dir_path.name)

        print(f"Loaded model name:\t{log_info['file_name']}")
        print(f"Start time:\t{log_info['start_time']}")
        print(f"End time:\t{log_info['end_time']}")

    return {"agents1":agents1, 
            "agents2":agents2,
            "model_dict1":model_dict1,
            "model_dict2":model_dict2,
        }


if __name__ == '__main__':
    # ------- configurations ------- #
    GlobalParams1=Params(param_type='k1',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=0.25,lr=0.0005, NumTrain=200)
    GlobalParams2=Params(param_type='k2',target_type='sigmoid',trick='clamp',loss_type='MSELoss',delta=0.03,w=0.25,lr=0.0005, NumTrain=200)
    agents1= Agents(GlobalParams1)
    agents2= Agents(GlobalParams2)
    print("On: ", GlobalParams1.device)

    parser=argparse.ArgumentParser()
    parser.add_argument('--load', type=bool, default=False, help='Load the model', required=False)
    args=parser.parse_args()

    configs= Config(GlobalParams1, GlobalParams2,
                    MaxEpoch=3,
                    OptimSteps=25)
    dir_path = Path(os.getcwd(),
                    'Results',
                    'BestModelsSaved',
                    f'{GlobalParams1.target_type}_{GlobalParams1.trick}_{GlobalParams1.lr}lr_{configs.MaxEpoch}epochs_{GlobalParams1.loss_type}_{(GlobalParams1.w)}w') # 0.25, 0.5, 0.75
    fig_path = Path(os.getcwd(),
                    'Results',
                    'Figs',
                    f'{GlobalParams1.target_type}_{GlobalParams1.trick}_{GlobalParams1.lr}lr_{configs.MaxEpoch}epochs_{GlobalParams1.loss_type}_{(GlobalParams1.w)}w') # 0.25, 0.5, 0.75

    if args.load ==False:
        agents1.reset_configs()
        agents2.reset_configs()
        optimizer, scheduler = configs.config_NN_params(agents1, agents2)
        assert (optimizer == configs.optimizer)
        assert (scheduler == configs.scheduler)
        # ------- train the model ------- #
        with torch.enable_grad():
            start_time = datetime.now().strftime('%B %d - %H:%M:%S')
            forward_losses, agents1, agents2, configs = train_loop(configs,agents1, agents2)
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
        save_the_model(dir_path, agents1, agents2, log_info)
    else:
        loaded_res=load_the_model(dir_path, configs)
        agents1=loaded_res['agents1']
        agents1=loaded_res['agents2']
        forward_losses = loaded_res['agents1'].loss
        

    # ------- plot and save ------- #
    if not fig_path.exists():
        fig_path.mkdir()

    with torch.inference_mode():
        plot = plot_results(pop1_dict=configs.pop1_dict, 
                            pop2_dict=configs.pop2_dict, 
                            loss=forward_losses, 
                            savefigs=True, to_path=fig_path)
        plot.FwdLoss(log=True)
        plot.Inventory_And_Price()
        plot.Decomposition_Inventory()
        plot.Key_Processes()
        plot.Terminal_Convergence()