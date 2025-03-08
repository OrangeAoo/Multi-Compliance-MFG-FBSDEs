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
def get_foward_loss(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, main_models}
   ## -------------------------------- Common Params -------------------------------- ##
  dB=pop1_dict['dB']
  dt=pop1_dict['GlobalParams'].dt
  NT=pop1_dict['GlobalParams'].NT
  NumTrain=pop1_dict['GlobalParams'].NumTrain
  device=pop1_dict['GlobalParams'].device
  trick=pop1_dict['GlobalParams'].trick
  ## -------------------------------- P1 Params -------------------------------- ##
  pi1=pop1_dict['GlobalParams'].pi
  h1=pop1_dict['GlobalParams'].h
  zeta1=pop1_dict['GlobalParams'].zeta
  beta1=pop1_dict['GlobalParams'].beta
  gamma1=pop1_dict['GlobalParams'].gamma
  sigma1=pop1_dict['GlobalParams'].sigma
  x1=pop1_dict['init_x']
  c1=pop1_dict['init_c']
  main_models1=pop1_dict['main_models']
  ## -------------------------------- P2 Params -------------------------------- ##
  pi2=pop2_dict['GlobalParams'].pi
  h2=pop2_dict['GlobalParams'].h
  zeta2=pop2_dict['GlobalParams'].zeta
  beta2=pop2_dict['GlobalParams'].beta
  gamma2=pop2_dict['GlobalParams'].gamma
  sigma2=pop2_dict['GlobalParams'].sigma
  x2=pop2_dict['init_x']
  c2=pop2_dict['init_c']
  main_models2=pop2_dict['main_models']
  
  if trick=='no':

    yc1=main_models1.yc0_model(x1).view(-1,1)
    yx1=main_models1.yx0_model(x1).view(-1,1)
    yc2=main_models2.yc0_model(x2).view(-1,1)
    yx2=main_models2.yx0_model(x2).view(-1,1)
    S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
    for j in range(1, NT+1):
        
        zx1 = main_models1.zx_models[j-1](x1)
        zc1=main_models1.zc_models[j-1](x1)
        zx2 = main_models2.zx_models[j-1](x2)
        zc2=main_models2.zc_models[j-1](x2)
        
        x1 =x1+ (h1-(1/zeta1+1/gamma1)*yx1+1/gamma1*S+c1)*dt+sigma1*dB[:,j].view(-1,1)
        x2 =x2+ (h2-(1/zeta1+1/gamma2)*yx2+1/gamma2*S+c2)*dt+sigma2*dB[:,j].view(-1,1)
        
        c1=c1-1/beta1*yc1*dt
        c2=c2-1/beta2*yc2*dt
        
        yc1=yc1-yx1*dt+zc1*dB[:,j].view(-1,1)
        yx1=yx1+zx1*dB[:,j].view(-1,1)
        yc2=yc2-yx2*dt+zc2*dB[:,j].view(-1,1)
        yx2=yx2+zx2*dB[:,j].view(-1,1)

        S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
        
    loss_yx1=torch.mean((yx1-target(x1,GlobalParams=pop1_dict['GlobalParams']))**2).to(device)
    loss_yc1=torch.mean((yc1)**2).to(device)
    loss_yx2=torch.mean((yx2-target(x2,GlobalParams=pop2_dict['GlobalParams']))**2).to(device)
    loss_yc2=torch.mean((yc2)**2).to(device)
    loss=(loss_yc1*100+loss_yx1)+(loss_yc2*100+loss_yx2)
    return loss.to(device)
  
  if trick=='logit': ## use yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde), dyx=-zx*(1+yx)*yx*dB

    yc1=main_models1.yc0_model(x1).view(-1,1)
    yx1_tilde=main_models1.yx0_model(x1).view(-1,1)
    yx1=-torch.sigmoid(yx1_tilde)  ## negative probability
    yc2=main_models2.yc0_model(x2).view(-1,1)
    yx2_tilde=main_models2.yx0_model(x2).view(-1,1)
    yx2=-torch.sigmoid(yx2_tilde)  ## negative probability
    S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
    for j in range(1, NT+1):
        
        zx1 = main_models1.zx_models[j-1](x1)
        zc1=main_models1.zc_models[j-1](x1)
        zx2 = main_models2.zx_models[j-1](x2)
        zc2=main_models2.zc_models[j-1](x2)
        
        x1 =x1+ (h1-(1/zeta1+1/gamma1)*yx1+1/gamma1*S+c1)*dt+sigma1*dB[:,j].view(-1,1)
        x2 =x2+ (h2-(1/zeta1+1/gamma2)*yx2+1/gamma2*S+c2)*dt+sigma2*dB[:,j].view(-1,1)
        
        c1=c1-1/beta1*yc1*dt
        c2=c2-1/beta2*yc2*dt
        
        yc1=yc1-yx1*dt+zc1*dB[:,j].view(-1,1)
        yx1_tilde=yx1_tilde-(zx1**2)*(yx1+1/2)*dt+zx1*dB[:,j].view(-1,1)
        yx1=-torch.sigmoid(yx1_tilde).view(-1,1)
        yc2=yc2-yx2*dt+zc2*dB[:,j].view(-1,1)
        yx2_tilde=yx2_tilde-(zx2**2)*(yx2+1/2)*dt+zx2*dB[:,j].view(-1,1)
        yx2=-torch.sigmoid(yx2_tilde).view(-1,1)

        S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
        
    loss_yx1=nn.BCEWithLogitsLoss()(yx1_tilde,-target(x1,GlobalParams=pop1_dict['GlobalParams'],target_type='indicator'))
    loss_yc1=torch.mean((yc1)**2).to(device)
    loss_yx2=nn.BCEWithLogitsLoss()(yx2_tilde,-target(x2,GlobalParams=pop2_dict['GlobalParams'],target_type='indicator'))
    loss_yc2=torch.mean((yc2)**2).to(device)
    loss=(loss_yc1*100+loss_yx1)+(loss_yc2*100+loss_yx2)
    return loss.to(device)
  
  if trick=='clamp': ## use dyx=-zx*(1+yx)*yx*dB ONLY

    yc1=main_models1.yc0_model(x1).view(-1,1)
    yx1=main_models1.yx0_model(x1).view(-1,1)
    yc2=main_models2.yc0_model(x2).view(-1,1)
    yx2=main_models2.yx0_model(x2).view(-1,1)
    S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
    for j in range(1, NT+1):
        
        zx1 = main_models1.zx_models[j-1](x1)
        zc1=main_models1.zc_models[j-1](x1)
        zx2 = main_models2.zx_models[j-1](x2)
        zc2=main_models2.zc_models[j-1](x2)
        
        x1 =x1+ (h1-(1/zeta1+1/gamma1)*yx1+1/gamma1*S+c1)*dt+sigma1*dB[:,j].view(-1,1)
        x2 =x2+ (h2-(1/zeta1+1/gamma2)*yx2+1/gamma2*S+c2)*dt+sigma2*dB[:,j].view(-1,1)
        
        c1=c1-1/beta1*yc1*dt
        c2=c2-1/beta2*yc2*dt
        
        yc1=yc1-yx1*dt+zc1*dB[:,j].view(-1,1)
        yx1=(yx1-zx1*(1+yx1)*yx1*dB[:,j].view(-1,1)).clamp(min=-1,max=0)
        yc2=yc2-yx2*dt+zc2*dB[:,j].view(-1,1)
        yx2=(yx2-zx2*(1+yx2)*yx2*dB[:,j].view(-1,1)).clamp(min=-1,max=0)

        S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
        
    loss_yx1=torch.mean((yx1-target(x1,GlobalParams=pop1_dict['GlobalParams']))**2).to(device)
    loss_yc1=torch.mean((yc1)**2).to(device)
    loss_yx2=torch.mean((yx2-target(x2,GlobalParams=pop2_dict['GlobalParams']))**2).to(device)
    loss_yc2=torch.mean((yc2)**2).to(device)
    loss=(loss_yc1*10+loss_yx1)+(loss_yc2*10+loss_yx2)
    return loss.to(device)

def get_target_path(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, main_models}
  ## -------------------------------- Common Params -------------------------------- ##
  dB=pop1_dict['dB']
  dt=pop1_dict['GlobalParams'].dt
  NT=pop1_dict['GlobalParams'].NT
  NumTrain=pop1_dict['GlobalParams'].NumTrain
  trick=pop1_dict['GlobalParams'].trick
  device=pop1_dict['GlobalParams'].device

  S_path=torch.ones(NT+1).to(device)
  ## -------------------------------- P1 Params -------------------------------- ##
  pi1=pop1_dict['GlobalParams'].pi
  h1=pop1_dict['GlobalParams'].h
  zeta1=pop1_dict['GlobalParams'].zeta
  beta1=pop1_dict['GlobalParams'].beta
  gamma1=pop1_dict['GlobalParams'].gamma
  sigma1=pop1_dict['GlobalParams'].sigma
  main_models1=pop1_dict['main_models']
  x1=pop1_dict['init_x']
  c1=pop1_dict['init_c']
  x1_path = torch.ones(NumTrain,NT+1).to(device)
  yx1_path = torch.ones(NumTrain,NT+1).to(device)
  yc1_path = torch.ones(NumTrain,NT+1).to(device)
  ## -------------------------------- P2 Params -------------------------------- ##
  pi2=pop2_dict['GlobalParams'].pi
  h2=pop2_dict['GlobalParams'].h
  zeta2=pop2_dict['GlobalParams'].zeta
  beta2=pop2_dict['GlobalParams'].beta
  gamma2=pop2_dict['GlobalParams'].gamma
  sigma2=pop2_dict['GlobalParams'].sigma
  main_models2=pop2_dict['main_models']
  x2=pop2_dict['init_x']
  c2=pop2_dict['init_c']
  x2_path = torch.ones(NumTrain,NT+1).to(device)
  yx2_path = torch.ones(NumTrain,NT+1).to(device)
  yc2_path = torch.ones(NumTrain,NT+1).to(device)
  
  with torch.inference_mode():

    if trick=='no':

      yc1=main_models1.yc0_model(x1).view(-1,1)
      yx1=main_models1.yx0_model(x1).view(-1,1)
      yc2=main_models2.yc0_model(x2).view(-1,1)
      yx2=main_models2.yx0_model(x2).view(-1,1)
      S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)

      x1_path[:,0] = x1.squeeze()
      yx1_path[:,0] = yx1.squeeze()
      yc1_path[:,0] = yc1.squeeze()
      x2_path[:,0] = x2.squeeze()
      yx2_path[:,0] = yx2.squeeze()
      yc2_path[:,0] = yc2.squeeze()
      S_path[0]=S

      for j in range(1, NT+1):
          
        zx1 = main_models1.zx_models[j-1](x1)
        zc1=main_models1.zc_models[j-1](x1)
        zx2 = main_models2.zx_models[j-1](x2)
        zc2=main_models2.zc_models[j-1](x2)
        
        x1 =x1+ (h1-(1/zeta1+1/gamma1)*yx1+1/gamma1*S+c1)*dt+sigma1*dB[:,j].view(-1,1)
        x2 =x2+ (h2-(1/zeta1+1/gamma2)*yx2+1/gamma2*S+c2)*dt+sigma2*dB[:,j].view(-1,1)
        
        c1=c1-1/beta1*yc1*dt
        c2=c2-1/beta2*yc2*dt
        
        yc1=yc1-yx1*dt+zc1*dB[:,j].view(-1,1)
        yx1=yx1+zx1*dB[:,j].view(-1,1)
        yc2=yc2-yx2*dt+zc2*dB[:,j].view(-1,1)
        yx2=yx2+zx2*dB[:,j].view(-1,1)

        S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)

        x1_path[:,j] = x1.squeeze()
        yx1_path[:,j] = yx1.squeeze()
        yc1_path[:,j] = yc1.squeeze()

        x2_path[:,j] = x2.squeeze()
        yx2_path[:,j] = yx2.squeeze()
        yc2_path[:,j] = yc2.squeeze()

        S_path[j]=S

    if trick=='logit': ## use yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde), dyx=-zx*(1+yx)*yx*dB

      yc1=main_models1.yc0_model(x1).view(-1,1)
      yx1_tilde=main_models1.yx0_model(x1).view(-1,1)
      yx1=-torch.sigmoid(yx1_tilde)  ## negative probability
      yc2=main_models2.yc0_model(x2).view(-1,1)
      yx2_tilde=main_models2.yx0_model(x2).view(-1,1)
      yx2=-torch.sigmoid(yx2_tilde)  ## negative probability
      S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
      
      x1_path[:,0] = x1.squeeze()
      yx1_path[:,0] = yx1.squeeze()
      yc1_path[:,0] = yc1.squeeze()
      x2_path[:,0] = x2.squeeze()
      yx2_path[:,0] = yx2.squeeze()
      yc2_path[:,0] = yc2.squeeze()
      S_path[0]=S

      for j in range(1, NT+1):
          
        zx1 = main_models1.zx_models[j-1](x1)
        zc1=main_models1.zc_models[j-1](x1)
        zx2 = main_models2.zx_models[j-1](x2)
        zc2=main_models2.zc_models[j-1](x2)
        
        x1 =x1+ (h1-(1/zeta1+1/gamma1)*yx1+1/gamma1*S+c1)*dt+sigma1*dB[:,j].view(-1,1)
        x2 =x2+ (h2-(1/zeta1+1/gamma2)*yx2+1/gamma2*S+c2)*dt+sigma2*dB[:,j].view(-1,1)
        
        c1=c1-1/beta1*yc1*dt
        c2=c2-1/beta2*yc2*dt
        
        yc1=yc1-yx1*dt+zc1*dB[:,j].view(-1,1)
        yx1_tilde=yx1_tilde-(zx1**2)*(yx1+1/2)*dt+zx1*dB[:,j].view(-1,1)
        yx1=-torch.sigmoid(yx1_tilde).view(-1,1)
        yc2=yc2-yx2*dt+zc2*dB[:,j].view(-1,1)
        yx2_tilde=yx2_tilde-(zx2**2)*(yx2+1/2)*dt+zx2*dB[:,j].view(-1,1)
        yx2=-torch.sigmoid(yx2_tilde).view(-1,1)

        S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
        
        x1_path[:,j] = x1.squeeze()
        yx1_path[:,j] = yx1.squeeze()
        yc1_path[:,j] = yc1.squeeze()

        x2_path[:,j] = x2.squeeze()
        yx2_path[:,j] = yx2.squeeze()
        yc2_path[:,j] = yc2.squeeze()

        S_path[j]=S

    if trick=='clamp': ## use dyx=-zx*(1+yx)*yx*dB ONLY

      yc1=main_models1.yc0_model(x1).view(-1,1)
      yx1=main_models1.yx0_model(x1).view(-1,1)
      yc2=main_models2.yc0_model(x2).view(-1,1)
      yx2=main_models2.yx0_model(x2).view(-1,1)
      S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
      
      x1_path[:,0] = x1.squeeze()
      yx1_path[:,0] = yx1.squeeze()
      yc1_path[:,0] = yc1.squeeze()
      x2_path[:,0] = x2.squeeze()
      yx2_path[:,0] = yx2.squeeze()
      yc2_path[:,0] = yc2.squeeze()
      S_path[0]=S
      
      for j in range(1, NT+1):
          
        zx1 = main_models1.zx_models[j-1](x1)
        zc1=main_models1.zc_models[j-1](x1)
        zx2 = main_models2.zx_models[j-1](x2)
        zc2=main_models2.zc_models[j-1](x2)
        
        x1 =x1+ (h1-(1/zeta1+1/gamma1)*yx1+1/gamma1*S+c1)*dt+sigma1*dB[:,j].view(-1,1)
        x2 =x2+ (h2-(1/zeta1+1/gamma2)*yx2+1/gamma2*S+c2)*dt+sigma2*dB[:,j].view(-1,1)
        
        c1=c1-1/beta1*yc1*dt
        c2=c2-1/beta2*yc2*dt
        
        yc1=yc1-yx1*dt+zc1*dB[:,j].view(-1,1)
        yx1=(yx1-zx1*(1+yx1)*yx1*dB[:,j].view(-1,1)).clamp(min=-1,max=0)
        yc2=yc2-yx2*dt+zc2*dB[:,j].view(-1,1)
        yx2=(yx2-zx2*(1+yx2)*yx2*dB[:,j].view(-1,1)).clamp(min=-1,max=0)

        S=(pi1*yx1.mean()/gamma1+pi2*yx2.mean()/gamma2)/(pi1/gamma1+pi2/gamma2)
        
        x1_path[:,j] = x1.squeeze()
        yx1_path[:,j] = yx1.squeeze()
        yc1_path[:,j] = yc1.squeeze()

        x2_path[:,j] = x2.squeeze()
        yx2_path[:,j] = yx2.squeeze()
        yc2_path[:,j] = yc2.squeeze()

        S_path[j] = S.squeeze()

  pop1_path_dict={'x':x1_path,
                  'yx':yx1_path,
                  'yc':yc1_path,
                  'S':S_path,
                  'expansion':-yc1_path/beta1,
                  'generation':-yx1_path/zeta1,
                  'trading':-(yx1_path-S)/gamma1,
                  'base':torch.FloatTensor([h1]*(NT+1)).to(device)}
  pop2_path_dict={'x':x2_path,
                  'yx':yx2_path,
                  'yc':yc2_path,
                  'S':S_path,
                  'expansion':-yc2_path/beta2,
                  'generation':-yx2_path/zeta2,
                  'trading':-(yx2_path-S)/gamma2,
                  'base':torch.FloatTensor([h2]*(NT+1)).to(device)}

  return pop1_path_dict, pop2_path_dict


class plot_results():
    def __init__(self,pop1_dict, pop2_dict, loss, PlotPaths=100, seed=42):#dB, init_x, init_c, GlobalParams, main_models, loss,PlotPaths=100, seed=42):
        ## -------------------------------- Common Params -------------------------------- ##
        self.loss=loss
        self.target_type='indicator' if (pop1_dict['GlobalParams'].target_type=='indicator' and pop1_dict['GlobalParams'].trick!='logit') else "sigmoid"
        self.delta=pop1_dict['GlobalParams'].delta
        self.K=pop1_dict['GlobalParams'].K
        self.NT=pop1_dict['GlobalParams'].NT
        self.dt=pop1_dict['GlobalParams'].dt
        self.t = np.array([i for i in range(self.NT+1)]) * self.dt
        self.NumTrain=pop1_dict['GlobalParams'].NumTrain
        self.number_of_paths=np.minimum(PlotPaths,self.NumTrain)
        self.seed=seed
        ## -------------------------------- P1 Params -------------------------------- ##
        x1=pop1_dict['init_x']
        c1=pop1_dict['init_c']
        main_models1=pop1_dict['main_models']
        self.GlobalParams1=pop1_dict['GlobalParams']
        self.h1=pop1_dict['GlobalParams'].h
        ## -------------------------------- P2 Params -------------------------------- ##
        x2=pop2_dict['init_x']
        c2=pop2_dict['init_c']
        main_models2=pop2_dict['main_models']
        self.GlobalParams2=pop2_dict['GlobalParams']
        self.h2=pop2_dict['GlobalParams'].h

        ## Processes And Rates
        self.pop1_path_dict,self.pop2_path_dict=get_target_path(pop1_dict, pop2_dict)
      
        torch.manual_seed(self.seed)
        idx_list = np.random.choice(self.NumTrain, self.number_of_paths, replace = False)
        self.pop1_plot = {'x':self.pop1_path_dict['x'].detach().numpy()[idx_list],
                          'yc':self.pop1_path_dict['yc'].detach().numpy()[idx_list],
                          'yx':self.pop1_path_dict['yx'].detach().numpy()[idx_list],
                          'S':self.pop1_path_dict['S'].detach().numpy(),
                          'base':self.pop1_path_dict['base'].detach().numpy(),
                          'expansion':self.pop1_path_dict['expansion'].detach().numpy()[idx_list],
                          'generation':self.pop1_path_dict['generation'].detach().numpy()[idx_list],
                          'trading':self.pop1_path_dict['trading'].detach().numpy()[idx_list]
                          }
        self.pop2_plot = {'x':self.pop2_path_dict['x'].detach().numpy()[idx_list],
                          'yc':self.pop2_path_dict['yc'].detach().numpy()[idx_list],
                          'yx':self.pop2_path_dict['yx'].detach().numpy()[idx_list],
                          'S':self.pop2_path_dict['S'].detach().numpy(),
                          'base':self.pop2_path_dict['base'].detach().numpy(),
                          'expansion':self.pop2_path_dict['expansion'].detach().numpy()[idx_list],
                          'generation':self.pop2_path_dict['generation'].detach().numpy()[idx_list],
                          'trading':self.pop2_path_dict['trading'].detach().numpy()[idx_list]}
           

    def FwdLoss(self,log=True):
        plt.figure(figsize=(10,6))
        plt.title("Forward_Loss vs Batch",fontsize=18)
        plt.plot(self.loss,label='P2',color='firebrick',alpha=0.6)
        plt.legend()

        if log==True:
            plt.yscale('log')

    def Integrate_Inventory(self): #,single_sample=False):
        plt.figure(figsize=(20,6))
        plt.subplot(131)
        plt.title("$X_t$")
        for i in range(self.number_of_paths):
          ax1,=plt.plot(self.t,self.pop1_plot['x'][i], color="green", alpha=0.3)
          ax2,=plt.plot(self.t,self.pop2_plot['x'][i], color="firebrick", alpha=0.3)
        plt.ylim(-1,1.5)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(132)
        plt.title("$Y^{X}_t$")
        for i in range(self.number_of_paths):
          ax1,=plt.plot(self.t,self.pop1_plot['yx'][i], color="green", alpha=0.3)
          ax2,=plt.plot(self.t,self.pop2_plot['yx'][i], color="firebrick", alpha=0.3)
        plt.ylim(-1,0.5)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(133)
        plt.title("$Y^{C}_t$")
        for i in range(self.number_of_paths):
          ax1,=plt.plot(self.t,self.pop1_plot['yc'][i], color="green", alpha=0.3)
          ax2,=plt.plot(self.t,self.pop2_plot['yc'][i], color="firebrick", alpha=0.3)
        plt.ylim(-1,0.5)
        plt.legend({'P1':ax1,'P2':ax2})
        
    
    def Decomposition_Inventory(self, cumulative=True, base_rate=False, market_price=False):
        
        if market_price==True:
          plt.figure(figsize=(6,4))
          plt.title("Price Process")
          plt.plot(self.t, self.pop1_plot['S'],color='darkgrey')
          plt.ylim(-1,0.5)

        plt.figure(figsize=(20,6) if base_rate==False else (20,4))
        if base_rate==True:
          plt.subplot(141)
          plt.title("Base Rate")
          ax1,=plt.plot(self.t,self.pop1_plot['base'], color="green")
          ax2,=plt.plot(self.t,self.pop2_plot['base'], color="firebrick")
          plt.ylim(-0.5,1) 
          plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(131 if base_rate==False else 142)
        plt.title("Expansion Rate")
        for i in range(self.number_of_paths):
          ax1,=plt.plot(self.t,self.pop1_plot['expansion'][i], color="green", alpha=0.3)
          ax2,=plt.plot(self.t,self.pop2_plot['expansion'][i], color="firebrick", alpha=0.3)
        plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(132 if base_rate==False else 143)
        plt.title("Generation Rate")
        for i in range(self.number_of_paths):
          ax1,=plt.plot(self.t,self.pop1_plot['generation'][i], color="green", alpha=0.3)
          ax2,=plt.plot(self.t,self.pop2_plot['generation'][i], color="firebrick", alpha=0.3)
        plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(133 if base_rate==False else 144)
        plt.title("Trading Rate")
        for i in range(self.number_of_paths):
          ax1,=plt.plot(self.t,self.pop1_plot['trading'][i], color="green", alpha=0.3)
          ax2,=plt.plot(self.t,self.pop2_plot['trading'][i], color="firebrick", alpha=0.3)
        plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        ## Accumulated Inventory - Decomposition
        if cumulative==True:
            self.pop1_plot['cum_expansion']=np.zeros(shape=(self.number_of_paths,1))
            self.pop1_plot['cum_expansion']=np.hstack([self.pop1_plot['cum_expansion'],self.dt*(self.pop1_plot['expansion'].cumsum(axis=1)[:,:-1])])
            self.pop2_plot['cum_expansion']=np.zeros(shape=(self.number_of_paths,1))
            self.pop2_plot['cum_expansion']=np.hstack([self.pop2_plot['cum_expansion'],self.dt*(self.pop2_plot['expansion'].cumsum(axis=1)[:,:-1])])
            
            self.pop1_plot['cum_generation']=np.zeros(shape=(self.number_of_paths,1))
            self.pop1_plot['cum_generation']=np.hstack([self.pop1_plot['cum_generation'],self.dt*(self.pop1_plot['generation'].cumsum(axis=1)[:,:-1])])
            self.pop2_plot['cum_generation']=np.zeros(shape=(self.number_of_paths,1))
            self.pop2_plot['cum_generation']=np.hstack([self.pop2_plot['cum_generation'],self.dt*(self.pop2_plot['generation'].cumsum(axis=1)[:,:-1])])
            
            self.pop1_plot['cum_trading']=np.zeros(shape=(self.number_of_paths,1))
            self.pop1_plot['cum_trading']=np.hstack([self.pop1_plot['cum_trading'],self.dt*(self.pop1_plot['trading'].cumsum(axis=1)[:,:-1])])
            self.pop2_plot['cum_trading']=np.zeros(shape=(self.number_of_paths,1))
            self.pop2_plot['cum_trading']=np.hstack([self.pop2_plot['cum_trading'],self.dt*(self.pop2_plot['trading'].cumsum(axis=1)[:,:-1])])
            
            self.pop1_plot['cum_base']=np.linspace(0,self.h1,self.NT+1)
            self.pop2_plot['cum_base']=np.linspace(0,self.h2,self.NT+1)

            plt.figure(figsize=(20,6) if base_rate==False else (20,4))

            if base_rate==True:
                plt.subplot(141)
                plt.title("Accumulated Baseline Growth")
                ax1,=plt.plot(self.t,self.pop1_plot['cum_base'], color="green")
                ax2,=plt.plot(self.t,self.pop2_plot['cum_base'], color="firebrick")
                plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(131 if base_rate==False else 142)
            plt.title("Accumulated Expansion")
            for i in range(self.number_of_paths):
              ax1,=plt.plot(self.t,self.pop1_plot['cum_expansion'][i], color="green", alpha=0.3)
              ax2,=plt.plot(self.t,self.pop2_plot['cum_expansion'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(132 if base_rate==False else 143)
            plt.title("Accumulated Generation")
            for i in range(self.number_of_paths):
              ax1,=plt.plot(self.t,self.pop1_plot['cum_generation'][i], color="green", alpha=0.3)
              ax2,=plt.plot(self.t,self.pop2_plot['cum_generation'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(133 if base_rate==False else 144)
            plt.title("Accumulated Trading")
            for i in range(self.number_of_paths):
              ax1,=plt.plot(self.t,self.pop1_plot['cum_trading'][i], color="green", alpha=0.3)
              ax2,=plt.plot(self.t,self.pop2_plot['cum_trading'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})


    def Terminal_Convergence(self,QQ_plot=True,Histogram=True):
        if QQ_plot==True:
          target_x1=target(self.pop1_path_dict['x'][:,-1],GlobalParams=self.GlobalParams1,target_type=self.target_type).detach()
          target_x2=target(self.pop2_path_dict['x'][:,-1],GlobalParams=self.GlobalParams2,target_type=self.target_type).detach()
          
          plt.figure(figsize=(9,4))

          plt.subplot(121)
          plt.title("QQ-Plot of Yx")
          ax1=plt.scatter(target_x1,self.pop1_path_dict['yx'][:,-1],s=3,alpha=0.3,color='green')
          ax2=plt.scatter(target_x2,self.pop2_path_dict['yx'][:,-1],s=3,alpha=0.3,color='firebrick')
          plt.plot(np.linspace(-1,1,5),np.linspace(-1,1,5),linestyle='--',linewidth=1,color='r')
          plt.legend({'P1':ax1,'P2':ax2})
          plt.xlabel("target(X)")
          plt.ylabel("Yx")

          plt.subplot(122)
          plt.title("QQ-Plot of Yc")
          ax1=plt.scatter(np.zeros_like(self.pop1_path_dict['yc'][:,-1]) ,self.pop1_path_dict['yc'][:,-1],s=3,alpha=0.3,color='green')
          ax2=plt.scatter(np.zeros_like(self.pop2_path_dict['yc'][:,-1]) ,self.pop2_path_dict['yc'][:,-1],s=3,alpha=0.3,color='firebrick')
          plt.hlines(y=0,xmin=-0.5,xmax=0.5,colors='r',linestyles='--')
          plt.legend({'P1':ax1,'P2':ax2})
          plt.xlim(-0.5,0.5)
          plt.ylim(-0.5,0.5)
          plt.xlabel("0's")
          plt.ylabel("Yc")

        if Histogram==True:  
          pop1_idx0=np.argwhere(np.where(self.pop1_path_dict['x'][:,-1]>self.K,1,0)).reshape(-1,)  # XT==0
          pop1_idx1=np.argwhere(np.where(self.pop1_path_dict['x'][:,-1]<self.K,1,0)).reshape(-1,)  # XT==1
          pop2_idx0=np.argwhere(np.where(self.pop2_path_dict['x'][:,-1]>self.K,1,0)).reshape(-1,)  # XT==0
          pop2_idx1=np.argwhere(np.where(self.pop2_path_dict['x'][:,-1]<self.K,1,0)).reshape(-1,)  # XT==1
          
          plt.figure(figsize=(9,9))

          plt.subplot(221)
          plt.title("Histogram of Yx - P1")
          sns.histplot(data=self.pop1_path_dict['yx'][pop1_idx0,-1], bins=100,stat='count',alpha=0.6,color= 'green')
          sns.histplot(data=self.pop1_path_dict['yx'][pop1_idx1,-1], bins=100,stat='count',alpha=0.3,color= 'grey')
          sns.kdeplot(self.pop1_path_dict['yx'][pop1_idx0,-1], color="green",label=f'$X_T$>{self.K},$Y_T\\rightarrow$ 0')
          sns.kdeplot(self.pop1_path_dict['yx'][pop1_idx1,-1], color="black",label=f'$X_T$<{self.K},$Y_T\\rightarrow$ -1')
          # plt.vlines(x=0,ymin=0,ymax=4,colors='r',linestyles='-.')
          plt.xlabel("Yx")
          plt.ylabel("Yx Count")
          plt.legend()

          plt.subplot(222)
          plt.title("Histogram of Yc - P1")
          sns.histplot(data=self.pop1_path_dict['yc'][:,-1], bins=100, stat='count',alpha= 0.5, color='green')
          sns.kdeplot(self.pop1_path_dict['yc'][:,-1], color="green")
          # plt.vlines(x=0,ymin=0,ymax=4,colors='r',linestyles='-.')
          plt.xlabel("Yc")
          plt.ylabel("Yc Count")

          plt.subplot(223)
          plt.title("Histogram of Yx - P2")
          sns.histplot(data=self.pop2_path_dict['yx'][pop2_idx0,-1], bins=100,stat='count',alpha=0.6,color= 'firebrick')
          sns.histplot(data=self.pop2_path_dict['yx'][pop2_idx1,-1], bins=100,stat='count',alpha=0.3,color= 'grey')
          sns.kdeplot(self.pop2_path_dict['yx'][pop2_idx0,-1], color="firebrick",label=f'$X_T$>{self.K},$Y_T\\rightarrow$ 0')
          sns.kdeplot(self.pop2_path_dict['yx'][pop2_idx1,-1], color="black",label=f'$X_T$<{self.K},$Y_T\\rightarrow$ -1')
          # plt.vlines(x=0,ymin=0,ymax=4,colors='r',linestyles='-.')
          plt.xlabel("Yx")
          plt.ylabel("Yx Count")
          plt.legend()

          plt.subplot(224)
          plt.title("Histogram of Yc - P2")
          sns.histplot(data=self.pop2_path_dict['yc'][:,-1], bins=100, stat='count',alpha= 0.5, color='firebrick')
          sns.kdeplot(self.pop2_path_dict['yc'][:,-1], color="firebrick")
          # plt.vlines(x=0,ymin=0,ymax=4,colors='r',linestyles='-.')
          plt.xlabel("Yc")
          plt.ylabel("Yc Count")


