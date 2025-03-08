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

from Model import *

forward_losses = []
    
## Functions
def Sample_Init(GlobalParams):
    '''
    Generate N samples of x0
    '''
    mean=GlobalParams.mean
    sd=GlobalParams.std
    N=GlobalParams.NumTrain
    device=GlobalParams.device
    xi = np.random.normal(mean,sd,size=N)
    
    return torch.FloatTensor(xi).view(-1,1).to(device)

def SampleBMIncr(GlobalParams):
  '''
  Returns Matrix of Dimension Npaths x Nsteps With Sample Increments of of BM
  Here an increment is of the form dB
  '''
  dt = GlobalParams.dt
  Npaths=GlobalParams.NumTrain
  Nsteps=GlobalParams.NT+1
  device=GlobalParams.device
  dB = np.sqrt(dt) * np.random.randn(Npaths, Nsteps)
  dB = torch.FloatTensor(dB).to(device)
  return dB

def target(x,GlobalParams,target_type=None):
  delta=GlobalParams.delta
  K=GlobalParams.K
  device=GlobalParams.device
  if target_type==None:
    target_type=GlobalParams.target_type

  if target_type=='sigmoid':# and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return -torch.sigmoid((K-x)/delta).to(device)
  
  if target_type=='indicator':# or (GlobalParams.target_type=='indicator' and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    x=x.detach().numpy()
    indicator=np.where(x<K,1,0)
    return -torch.FloatTensor(indicator).to(device)
  
  if target_type=='original':# or (GlobalParams.target_type=='original' and GlobalParams.trick=='logit'):
    return torch.FloatTensor((K-x)/delta).to(device)
  
  else:
      print("Please check whether 'target_type' matches 'logit_trick' :)")

# Forward Loss
def get_foward_loss(dB, init_x,init_c, GlobalParams, main_models):
  h=GlobalParams.h
  zeta=GlobalParams.zeta
  beta=GlobalParams.beta
  gamma=GlobalParams.gamma
  sigma=GlobalParams.sigma
  device=GlobalParams.device
  dt=GlobalParams.dt
  NT=GlobalParams.NT
  NumTrain=GlobalParams.NumTrain

  x = init_x
  c = init_c
  
  if GlobalParams.trick=='no':

    yc=main_models.yc0_model(x)
    yx=main_models.yx0_model(x)
    for j in range(1, NT+1):
        
        zx = main_models.zx_models[j-1](x)
        zc=main_models.zc_models[j-1](x)
        
        x =x+ (h-(1/zeta+1/gamma)*yx+1/gamma*yx.mean()+c)*dt+sigma*dB[:,j].view(-1,1)
        c=c-1/beta*yc*dt
        yc=yc-yx*dt+zc*dB[:,j].view(-1,1)
        yx=yx+zx*dB[:,j].view(-1,1)
        
    loss_yx=torch.mean((yx-target(x,GlobalParams=GlobalParams))**2).to(device)
    # loss_yx=GlobalParams.Loss_type(yx,)
    loss_yc=torch.mean((yc)**2).to(device)
    loss=loss_yc*100+loss_yx
    return loss.to(device)
  
  if GlobalParams.trick=='logit': ## use yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde), dyx=-zx*(1+yx)*yx*dB
    
    yc=main_models.yc0_model(x)
    yx_tilde=main_models.yx0_model(x)
    yx=(-torch.sigmoid(yx_tilde)).clamp(min=-1,max=0)  ## negative probability
    for j in range(1, NT+1):
      
      zx = main_models.zx_models[j-1](x)
      zc=main_models.zc_models[j-1](x)
      
      x =x+ (h-(1/zeta+1/gamma)*yx+1/gamma*yx.mean()+c)*dt+sigma*dB[:,j].view(-1,1)
      c=c-1/beta*yc*dt
      yc=yc-yx*dt+zc*dB[:,j].view(-1,1)
      # yx=yx+zx*dB[:,j].view(-1,1)
      yx_tilde=(yx_tilde-(zx**2)*(yx+1/2)*dt+zx*dB[:,j].view(-1,1))
      yx=(-torch.sigmoid(yx_tilde)).clamp(min=-1,max=0)
    # loss_yx_tilde=torch.mean((yx_tilde-target(x,GlobalParams=GlobalParams))**2)
    loss_yx_tilde=nn.BCEWithLogitsLoss()(yx_tilde,-target(x,GlobalParams=GlobalParams,target_type='indicator'))
    loss_yc=torch.mean((yc)**2)
    loss=loss_yc*1000+loss_yx_tilde
    return loss.to(device)
  
  if GlobalParams.trick=='clamp': ## use dyx=-zx*(1+yx)*yx*dB ONLY
    
    yc=main_models.yc0_model(x)
    yx=main_models.yx0_model(x)
    for j in range(1, NT+1):
      
      zx = main_models.zx_models[j-1](x)
      zc=main_models.zc_models[j-1](x)
      x =x+ (h-(1/zeta+1/gamma)*yx+1/gamma*yx.mean()+c)*dt+sigma*dB[:,j].view(-1,1)
      c=c-1/beta*yc*dt
      yc=yc-yx*dt+zc*dB[:,j].view(-1,1)
      yx=(yx-zx*(1+yx)*yx*dB[:,j].view(-1,1)).clamp(min=-1,max=0)
      # print(yx.shape)
    # print(torch.min(yx),' ~ ',torch.max(yx))
    loss_yx=torch.mean((yx-target(x,GlobalParams=GlobalParams))**2)
    loss_yc=torch.mean((yc)**2)
    loss=loss_yc*100+loss_yx
    return loss.to(device)

def get_target_path(dB, init_x, init_c, GlobalParams, main_models):
  h=GlobalParams.h
  zeta=GlobalParams.zeta
  beta=GlobalParams.beta
  gamma=GlobalParams.gamma
  sigma=GlobalParams.sigma
  device=GlobalParams.device
  dt=GlobalParams.dt
  NT=GlobalParams.NT
  NumTrain=GlobalParams.NumTrain

  x = init_x
  c = init_c
  x_path = torch.ones(NumTrain,NT+1)
  yx_path = torch.ones(NumTrain,NT+1)
  yc_path = torch.ones(NumTrain,NT+1)

  with torch.inference_mode():

    if GlobalParams.trick=='no':

      yc=main_models.yc0_model(x)
      yx=main_models.yx0_model(x)

      x_path[:,0] = x.squeeze()
      yx_path[:,0] = yx.squeeze()
      yc_path[:,0] = yc.squeeze()
      for j in range(1, NT+1):
          
        zx = main_models.zx_models[j-1](x)
        zc=main_models.zc_models[j-1](x)
        x =x+ (h-(1/zeta+1/gamma)*yx+1/gamma*yx.mean()+c)*dt+sigma*dB[:,j].view(-1,1)
        c=c-1/beta*yc*dt
        yc=yc-yx*dt+zc*dB[:,j].view(-1,1)
        yx=yx+zx*dB[:,j].view(-1,1)
        x_path[:,j] = x.squeeze()
        yx_path[:,j] = yx.squeeze()
        yc_path[:,j] = yc.squeeze()
  
  
    if GlobalParams.trick=='logit': ## use yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde), dyx=-zx*(1+yx)*yx*dB
      
      yc=main_models.yc0_model(x)
      yx_tilde=main_models.yx0_model(x)
      yx=(-torch.sigmoid(yx_tilde)).clamp(min=-1,max=0)

      x_path[:,0] = x.squeeze()
      yx_path[:,0] = yx.squeeze()
      yc_path[:,0] = yc.squeeze()
      for j in range(1, NT+1):
        
        zx = main_models.zx_models[j-1](x)
        zc=main_models.zc_models[j-1](x)
        
        x =x+ (h-(1/zeta+1/gamma)*yx+1/gamma*yx.mean()+c)*dt+sigma*dB[:,j].view(-1,1)
        c=c-1/beta*yc*dt
        yc=yc-yx*dt+zc*dB[:,j].view(-1,1)
        # yx=yx+zx*dB[:,j].view(-1,1)
        yx_tilde=(yx_tilde-(zx**2)*(yx+1/2)*dt+zx*dB[:,j].view(-1,1))
        yx=-torch.sigmoid(yx_tilde).clamp(min=-1,max=0)
        x_path[:,j] = x.squeeze()
        yx_path[:,j] = yx.squeeze()
        yc_path[:,j] = yc.squeeze()

    if GlobalParams.trick=='clamp': ## use dyx=-zx*(1+yx)*yx*dB ONLY
      
      yc=main_models.yc0_model(x)
      yx=main_models.yx0_model(x)

      x_path[:,0] = x.squeeze()
      yx_path[:,0] = yx.squeeze()
      yc_path[:,0] = yc.squeeze()
      for j in range(1, NT+1):
        
        zx = main_models.zx_models[j-1](x)
        zc=main_models.zc_models[j-1](x)
        x =x+ (h-(1/zeta+1/gamma)*yx+1/gamma*yx.mean()+c)*dt+sigma*dB[:,j].view(-1,1)
        c=c-1/beta*yc*dt
        yc=yc-yx*dt+zc*dB[:,j].view(-1,1)
        yx=(yx-zx*(1+yx)*yx*dB[:,j].view(-1,1)).clamp(min=-1,max=0)
        x_path[:,j] = x.squeeze()
        yx_path[:,j] = yx.squeeze()
        yc_path[:,j] = yc.squeeze()
  return x_path.detach(), yx_path.detach(), yc_path.detach()


class plot_results():
    def __init__(self,dB, init_x, init_c, GlobalParams, main_models, loss,PlotPaths=100, seed=42):
        self.GlobalParams=GlobalParams
        self.main_models=main_models
        self.delta=GlobalParams.delta
        self.K=GlobalParams.K
        self.NT=GlobalParams.NT
        self.dt=GlobalParams.dt
        self.NumTrain=GlobalParams.NumTrain
        self.loss=loss
        self.number_of_paths=np.minimum(PlotPaths,self.NumTrain)
        self.seed=seed
        ## Processes
        self.x_path,self.yx_path, self.yc_path=get_target_path(dB, init_x=init_x,init_c=init_c,
                                                                       GlobalParams=GlobalParams,
                                                                       main_models=main_models)                                             
        self.expansion_rate=-self.yc_path/self.GlobalParams.beta
        self.generation_rate=-self.yx_path/self.GlobalParams.zeta
        self.trading_rate=-(self.yx_path-torch.mean(self.yx_path,axis=0))/self.GlobalParams.gamma
        
        torch.manual_seed(self.seed)
        idx_list = np.random.choice(self.NumTrain, self.number_of_paths, replace = False)
        self.x_plot = self.x_path.detach().numpy()[idx_list]
        self.yx_plot = self.yx_path.detach().numpy()[idx_list]
        self.yc_plot = self.yc_path.detach().numpy()[idx_list]
        self.expansion_rate_plot=self.expansion_rate.detach().numpy()[idx_list]
        self.generation_rate_plot=self.generation_rate.detach().numpy()[idx_list]
        self.trading_rate_plot=self.trading_rate.detach().numpy()[idx_list]
        
        self.t = np.array([i for i in range(self.NT+1)]) * self.dt



    def FwdLoss(self,log=True):
        plt.figure(figsize=(10,6))
        plt.title("Forward_Loss vs Batch",fontsize=18)
        plt.plot(self.loss)

        if log==True:
            plt.yscale('log')

    def Integrate_Inventory(self,single_sample=False):
        plt.figure(figsize=(20,6))
        plt.subplot(131)
        plt.title("$X_t$")
        for i in range(self.number_of_paths):
                plt.plot(self.t,self.x_plot[i], color="darkgreen", alpha=0.5)
        plt.ylim(-1,1.5)

        plt.subplot(132)
        plt.title("$Y^{X}_t$")
        for i in range(self.number_of_paths):
                plt.plot(self.t,self.yx_plot[i], color="darkgreen", alpha=0.5)
        plt.ylim(-1,0.5)

        plt.subplot(133)
        plt.title("$Y^{C}_t$")
        for i in range(self.number_of_paths):
                plt.plot(self.t,self.yc_plot[i], color="darkgreen", alpha=0.5)
        plt.ylim(-1,0.5)

        ## Integrated Plot -- a single sample
        if single_sample==True:
            torch.manual_seed(self.seed)
            idx=random.randint(0,self.number_of_paths)
            plt.figure(figsize=(10,8))
            plt.subplot()
            plt.plot(self.t,self.x_plot[idx], color="blue", alpha=0.5,label='X')
            plt.plot(self.t,self.yx_plot[idx], color="black", linestyle='--',alpha=0.5,label="$Y^{X}$")
            plt.plot(self.t,self.yc_plot[idx], color="orange", linestyle='--',alpha=0.5,label="$Y^{C}$")
          
            plt.hlines(y=[0,-1],xmin=0,xmax=1,colors='firebrick',linestyles='-.')
            plt.title("Comparison of A Particular Path")
            plt.legend()
        
    
    def Decomposition_Inventory(self, cumulative=True, base_rate=False):
        plt.figure(figsize=(20,6) if base_rate==False else (20,4))

        if base_rate==True:
          plt.subplot(141)
          plt.title("Base Rate")
          for i in range(self.number_of_paths):
              plt.plot(self.t,[self.GlobalParams.h]*len(self.t), color="darkgreen", alpha=0.5)

        plt.subplot(131 if base_rate==False else 142)
        plt.title("Expansion Rate")
        for i in range(self.number_of_paths):
            plt.plot(self.t,self.expansion_rate_plot[i], color="darkgreen", alpha=0.5)
        plt.ylim(-0.5,1)

        plt.subplot(132 if base_rate==False else 143)
        plt.title("Generation Rate")
        for i in range(self.number_of_paths):
            plt.plot(self.t,self.generation_rate_plot[i], color="darkgreen", alpha=0.5)
        plt.ylim(-0.5,1)

        plt.subplot(133 if base_rate==False else 144)
        plt.title("Trading Rate")
        for i in range(self.number_of_paths):
            plt.plot(self.t,self.trading_rate_plot[i], color="darkgreen", alpha=0.5)
        plt.ylim(-0.5,1)

        ## Accumulated Inventory - Decomposition
        if cumulative==True:
            self.cum_expansion_plot=np.zeros(shape=(self.number_of_paths,1))
            self.cum_expansion_plot=np.hstack([self.cum_expansion_plot,self.dt*(self.expansion_rate_plot.cumsum(axis=1)[:,:-1])])
            
            self.cum_generation_plot=np.zeros(shape=(self.number_of_paths,1))
            self.cum_generation_plot=np.hstack([self.cum_generation_plot,self.dt*(self.generation_rate_plot.cumsum(axis=1)[:,:-1])])
            
            self.cum_trading_plot=np.zeros(shape=(self.number_of_paths,1))
            self.cum_trading_plot=np.hstack([self.cum_trading_plot,self.dt*(self.trading_rate_plot.cumsum(axis=1)[:,:-1])])
            
            self.cum_base_plot=np.linspace(0,self.GlobalParams.h,self.NT+1)

            plt.figure(figsize=(20,6) if base_rate==False else (20,4))

            if base_rate==True:
                plt.subplot(141)
                plt.title("Accumulated Baseline Growth")
                for i in range(self.number_of_paths):
                    plt.plot(self.t,self.cum_base_plot, color="darkgreen", alpha=0.5)

            plt.subplot(131 if base_rate==False else 142)
            plt.title("Accumulated Expansion")
            for i in range(self.number_of_paths):
                plt.plot(self.t,self.cum_expansion_plot[i], color="darkgreen", alpha=0.5)

            plt.subplot(132 if base_rate==False else 143)
            plt.title("Accumulated Generation")
            for i in range(self.number_of_paths):
                plt.plot(self.t,self.cum_generation_plot[i], color="darkgreen", alpha=0.5)

            plt.subplot(133 if base_rate==False else 144)
            plt.title("Accumulated Trading")
            for i in range(self.number_of_paths):
                plt.plot(self.t,self.cum_trading_plot[i], color="darkgreen", alpha=0.5)
          

    def Terminal_Convergence(self,QQ_plot=True,Histogram=True):
        if QQ_plot==True:
          plt.figure(figsize=(9,4))

          plt.subplot(121)
          plt.title("QQ-Plot of Yx")
          target_type='indicator' if (self.GlobalParams.target_type=='indicator' and self.GlobalParams.trick!='logit') else "sigmoid"
          target_x=target(self.x_path[:,-1],GlobalParams=self.GlobalParams,target_type=target_type).detach()
          plt.scatter(target_x,self.yx_path[:,-1],s=3)
          plt.plot(np.linspace(-1,1,5),np.linspace(-1,1,5),linestyle='--',linewidth=1,color='r')
          plt.xlabel("target(X)")
          plt.ylabel("Yx")

          plt.subplot(122)
          plt.title("QQ-Plot of Yc")
          plt.scatter(np.zeros_like(self.yc_path[:,-1]) ,self.yc_path[:,-1],s=3)
          plt.hlines(y=0,xmin=-0.5,xmax=0.5,colors='r',linestyles='-.')
          plt.xlim(-0.5,0.5)
          plt.ylim(-0.5,0.5)
          plt.xlabel("0's")
          plt.ylabel("Yc")

        if Histogram==True:  
          idx0=np.argwhere(np.where(self.x_path[:,-1]>self.K,1,0)).reshape(-1,)  # XT==0
          idx1=np.argwhere(np.where(self.x_path[:,-1]<self.K,1,0)).reshape(-1,)  # XT==1
          
          plt.figure(figsize=(9,4))

          plt.subplot(121)
          plt.title("Histogram of Yx")
          sns.histplot(data=self.yx_path[idx0,-1], bins=100,stat='count',alpha=0.7,color= 'grey')
          sns.histplot(data=self.yx_path[idx1,-1], bins=100,stat='count',alpha=0.3,color= 'green')
          sns.kdeplot(self.yx_path[idx0,-1], color="black",label=f'$X_T$>{self.K},$Y_T\\rightarrow$ 0')
          sns.kdeplot(self.yx_path[idx1,-1], color="green",label=f'$X_T$<{self.K},$Y_T\\rightarrow$ -1')
          # plt.vlines(x=0,ymin=0,ymax=4,colors='r',linestyles='-.')
          plt.xlabel("Yx")
          plt.ylabel("Yx Count")
          plt.legend()

          plt.subplot(122)
          plt.title("Histogram of Yc")
          sns.histplot(data=self.yc_path[:,-1], bins=100, stat='count',alpha= 0.7, color='grey')
          sns.kdeplot(self.yc_path[:,-1], color="black")
          # plt.vlines(x=0,ymin=0,ymax=4,colors='r',linestyles='-.')
          plt.xlabel("Yc")
          plt.ylabel("Yc Count")


