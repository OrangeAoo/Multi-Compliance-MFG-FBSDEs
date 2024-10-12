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
  Generate N samples of x0.
  '''
  mean=GlobalParams.mean
  sd=GlobalParams.std
  N=GlobalParams.NumTrain
  device=GlobalParams.device
  xi = np.random.normal(mean,sd,size=N)
  
  return torch.FloatTensor(xi).view(-1,1).to(device)

def SampleBMIncr(GlobalParams):
  '''
  Returns Matrix of Dimension Npaths x Nsteps With Sample Increments of of BM.
  Here an increment is of the form dB.
  '''
  dt = GlobalParams.dt
  Npaths=GlobalParams.NumTrain
  Nsteps=GlobalParams.NT2+1
  device=GlobalParams.device
  dB = np.sqrt(dt) * np.random.randn(Npaths, Nsteps)
  dB = torch.FloatTensor(dB).to(device)
  return dB

def target_V(x_t1,GlobalParams,target_type=None):
  '''
  Terminal target of V given the terminal values of X (with specified target_type). 
  '''
  delta=GlobalParams.delta
  K=GlobalParams.K
  device=GlobalParams.device
  if target_type==None:
    target_type=GlobalParams.target_type

  if target_type=='sigmoid':# and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (torch.sigmoid((K-x_t1)/delta)).to(device)
  
  if target_type=='indicator':# or (GlobalParams.target_type=='indicator' and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (torch.where(x_t1<K,1.0,0.0)).to(device)
  
  else:
      print("Please check whether 'target_type' matches 'trick' :)")

def target_U(x_t1,y_t1,GlobalParams,target_type=None):
  '''
  Terminal target of U given the terminal values of X (with specified target_type). 
  '''
  delta=GlobalParams.delta
  K=GlobalParams.K
  device=GlobalParams.device
  if target_type==None:
    target_type=GlobalParams.target_type

  if target_type=='sigmoid':# and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (y_t1*torch.sigmoid((x_t1-K)/delta)).to(device)
  
  if target_type=='indicator':# or (GlobalParams.target_type=='indicator' and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (y_t1*torch.where(x_t1>K,1.0,0.0)).to(device)
 
  else:
      print("Please check if 'target_type' matches 'trick' :)")

def target_Y(x_t2,GlobalParams,target_type=None):
  '''
  Terminal target of Y given the terminal values of X (with specified target_type). 
  '''
  delta=GlobalParams.delta
  K=GlobalParams.K
  device=GlobalParams.device
  if target_type==None:
    target_type=GlobalParams.target_type

  if target_type=='sigmoid':# and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (torch.sigmoid((K-x_t2)/delta)).to(device)
  
  if target_type=='indicator':# or (GlobalParams.target_type=='indicator' and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (torch.where(x_t2<K,1.0,0.0)).to(device)

  else:
      print("Please check whether 'target_type' matches 'trick' :)")

# Calculate Loss
def Loss(pred,targ,GlobalParams, loss_type=None):
  ''' 
  Customized loss function with specified loss_type e.g. MSELoss, BCELoss, etc. 
  '''
  device=GlobalParams.device
  if loss_type==None:
    loss_type=GlobalParams.loss_type
  if loss_type=='MSELoss':
     return torch.mean((pred-targ)**2).to(device)
  if loss_type=='BCELoss':
     return nn.BCELoss()(pred,targ).to(device)
  if loss_type=='BCEWithLogitsLoss':
     return nn.BCEWithLogitsLoss()(pred,targ).to(device)
  print(f"{loss_type} is wrong.\nPlease check the if 'loss_type' is specified correctly :)")
    
# Forward Loss
def get_foward_loss(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, main_models}
  '''
  Perform the stepwise approximation for a single iteration. 
  '''
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
  dB1=pop1_dict['dB']
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
  dB2=pop2_dict['dB']
  ## -------------------------------- Common Params -------------------------------- ##
  w=pop1_dict['GlobalParams'].w
  K=pop1_dict['GlobalParams'].K
  dt=pop1_dict['GlobalParams'].dt
  NT1=pop1_dict['GlobalParams'].NT1
  NT2=pop1_dict['GlobalParams'].NT2
  NumTrain=pop1_dict['GlobalParams'].NumTrain
  device=pop1_dict['GlobalParams'].device
  trick=pop1_dict['GlobalParams'].trick
  loss_type=pop1_dict['GlobalParams'].loss_type
  omiga1=(pi1/gamma1)/(pi1/gamma1+pi2/gamma2)
  omiga2=(pi2/gamma2)/(pi1/gamma1+pi2/gamma2)
  
  if trick=='logit': ## use yx_tilde=logit(yx), yx=sigmoid(yx_tilde), dyx=zx*(1-yx)*yx*dB
    for j in range(0, NT2+1):
      if j==0:  # @ 0 --> init for Prd1&2
        v1_tilde=w*main_models1.v0_model(x1).view(-1,1)
        v2_tilde=w*main_models2.v0_model(x2).view(-1,1)
        v1=w*torch.sigmoid(v1_tilde/w).view(-1,1).to(device)
        v2=w*torch.sigmoid(v2_tilde/w).view(-1,1).to(device)

        u1_tilde=w*main_models1.u0_model(x1).view(-1,1)
        u2_tilde=w*main_models2.u0_model(x2).view(-1,1)
        u1=w*torch.sigmoid(u1_tilde/w).view(-1,1).to(device)
        u2=w*torch.sigmoid(u2_tilde/w).view(-1,1).to(device)

        y1_tilde=w*main_models1.y0_model(x1).view(-1,1)
        y2_tilde=w*main_models2.y0_model(x2).view(-1,1)
        y1=w*torch.sigmoid(y1_tilde/w).view(-1,1).to(device)
        y2=w*torch.sigmoid(y2_tilde/w).view(-1,1).to(device)
        
      else:
        x1 = x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
        x2 = x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

        c1=c1+a1*dt
        c2=c2+a2*dt

        if j<=NT1:
          zv1 = main_models1.zv_models[j-1](x1)
          zv2 = main_models2.zv_models[j-1](x2)

          zu1 = main_models1.zu_models[j-1](x1)
          zu2 = main_models2.zu_models[j-1](x2)
          
        zy1 = main_models1.zy_models[j-1](x1)
        zy2 = main_models2.zy_models[j-1](x2)
        
        v1_tilde=v1_tilde+(zv1**2)*(v1-w/2)*dt+w*zv1*dB1[:,j].view(-1,1) if j<=NT1 else v1_tilde
        v2_tilde=v2_tilde+(zv2**2)*(v2-w/2)*dt+w*zv2*dB2[:,j].view(-1,1) if j<=NT1 else v2_tilde
        v1=w*torch.sigmoid(v1_tilde/w).view(-1,1).to(device) if j<=NT1 else v1
        v2=w*torch.sigmoid(v2_tilde/w).view(-1,1).to(device) if j<=NT1 else v2

        u1_tilde=u1_tilde+(zu1**2)*(u1-w/2)*dt+w*zu1*dB1[:,j].view(-1,1) if j<=NT1 else u1_tilde
        u2_tilde=u2_tilde+(zu2**2)*(u2-w/2)*dt+w*zu2*dB2[:,j].view(-1,1) if j<=NT1 else u2_tilde
        u1=w*torch.sigmoid(u1_tilde/w).view(-1,1).to(device) if j<=NT1 else u1
        u2=w*torch.sigmoid(u2_tilde/w).view(-1,1).to(device) if j<=NT1 else u2

        y1_tilde=y1_tilde+(zy1**2)*(y1-w/2)*dt+w*zy1*dB1[:,j].view(-1,1)
        y2_tilde=y2_tilde+(zy2**2)*(y2-w/2)*dt+w*zy2*dB2[:,j].view(-1,1)
        y1=(w*torch.sigmoid(y1_tilde/w).view(-1,1)).to(device)
        y2=(w*torch.sigmoid(y2_tilde/w).view(-1,1)).to(device)
        
      S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
      
      g1=(v1+u1)/zeta1 if j<=NT1 else y1/zeta1
      g2=(v2+u2)/zeta2 if j<=NT1 else y2/zeta2

      Gamma1=(v1+u1-S)/gamma1 if j<=NT1 else (y1-S)/gamma1
      Gamma2=(v2+u2-S)/gamma2 if j<=NT1 else (y2-S)/gamma2

      a1=((NT1-j)*(v1+u1)+(NT2-NT1)*y1)*dt/beta1 if j<=NT1 else (NT2-j)*y1*dt/beta1  #*dt= 0.02
      a2=((NT1-j)*(v2+u2)+(NT2-NT1)*y2)*dt/beta2 if j<=NT1 else (NT2-j)*y2*dt/beta2

      if j==NT1: # @ NT1 --> Loss for Prd1: loss_v, loss_u
        x1_t1, x2_t1, y1_t1, y2_t1 = x1, x2, y1, y2 
    
        loss_v1=Loss(pred=v1_tilde/w,targ=target_V(x_t1=x1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
        loss_v2=Loss(pred=v2_tilde/w,targ=target_V(x_t1=x2,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
        
        loss_u1=Loss(pred=u1_tilde/w,targ=target_U(x_t1=x1,y_t1=y1/w,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='MSELoss')
        loss_u2=Loss(pred=u2_tilde/w,targ=target_U(x_t1=x2,y_t1=y2/w,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='MSELoss')
        
        x1 = nn.ReLU()(x1-K)  # @ NT1: hand in min(K,xt1)
        x2 = nn.ReLU()(x2-K)
    # @ NT2 --> Loss for Prd2: loss_y
    loss_y1=Loss(pred=y1_tilde/w,targ=target_Y(x_t2=x1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
    loss_y2=Loss(pred=y2_tilde/w,targ=target_Y(x_t2=x2,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
    
    loss=(loss_v1+loss_u1+loss_y1)+(loss_v2+loss_u2+loss_y2)
    return loss.to(device)
  
  if trick=='clamp':  ## use dy=-zy*(1-y)*y*dB ONLY
    for j in range(0, NT2+1):
      if j==0:  # @ 0 --> init for Prd1&2
        v1=w*main_models1.v0_model(x1).view(-1,1)#.clamp(min=0,max=w)
        v2=w*main_models2.v0_model(x2).view(-1,1)#.clamp(min=0,max=w)
        u1=w*main_models1.u0_model(x1).view(-1,1)#.clamp(min=0,max=w)
        u2=w*main_models2.u0_model(x2).view(-1,1)#.clamp(min=0,max=w)
        
        y1=w*main_models1.y0_model(x1).view(-1,1)#.clamp(min=0,max=w)
        y2=w*main_models2.y0_model(x2).view(-1,1)#.clamp(min=0,max=w)
        
        S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
      else:
        x1 =x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
        x2 =x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

        c1=c1+a1*dt
        c2=c2+a2*dt

        if j<=NT1: 
          zv1 = main_models1.zv_models[j-1](x1)
          zv2 = main_models2.zv_models[j-1](x2)

          zu1 = main_models1.zu_models[j-1](x1)
          zu2 = main_models2.zu_models[j-1](x2)
          
        zy1 = main_models1.zy_models[j-1](x1)
        zy2 = main_models2.zy_models[j-1](x2)
        
        v1=((v1+v1*(1-v1/w)*zv1*dB1[:,j].view(-1,1)) if j<=NT1 else v1).clamp(min=0,max=w)
        v2=((v2+v2*(1-v2/w)*zv2*dB2[:,j].view(-1,1)) if j<=NT1 else v2).clamp(min=0,max=w)

        u1=((u1+u1*(1-u1/w)*zu1*dB1[:,j].view(-1,1)) if j<=NT1 else u1).clamp(min=0,max=w)
        u2=((u2+u2*(1-u2/w)*zu2*dB2[:,j].view(-1,1)) if j<=NT1 else u2).clamp(min=0,max=w)

        y1=((y1+y1*(1-y1/w)*zy1*dB1[:,j].view(-1,1))).clamp(min=0,max=w)
        y2=((y2+y2*(1-y2/w)*zy2*dB2[:,j].view(-1,1))).clamp(min=0,max=w)

        S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
      
      g1=(v1+u1)/zeta1 if j<=NT1 else y1/zeta1
      g2=(v2+u2)/zeta2 if j<=NT1 else y2/zeta2

      Gamma1=(v1+u1-S)/gamma1 if j<=NT1 else (y1-S)/gamma1
      Gamma2=(v2+u2-S)/gamma2 if j<=NT1 else (y2-S)/gamma2

      a1=((NT1-j)*(v1+u1)+(NT2-NT1)*y1)*dt/beta1 if j<=NT1 else (NT2-j)*dt*y1/beta1
      a2=((NT1-j)*(v2+u2)+(NT2-NT1)*y2)*dt/beta2 if j<=NT1 else (NT2-j)*dt*y2/beta2

      if j==NT1:  # @ NT1 --> Loss for Prd1: loss_v, loss_u; Clearance: x <- relu(x-k), hand in min(x,K)
        x1_t1, x2_t1, y1_t1, y2_t1 = x1, x2, y1, y2 

        loss_v1=Loss(pred=v1/w,targ=target_V(x_t1=x1_t1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type=loss_type)
        loss_v2=Loss(pred=v2/w,targ=target_V(x_t1=x2_t1,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type=loss_type)
        
        loss_u1=Loss(pred=u1/w,targ=target_U(x_t1=x1_t1,y_t1=y1_t1/w,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='MSELoss')
        loss_u2=Loss(pred=u2/w,targ=target_U(x_t1=x2_t1,y_t1=y2_t1/w,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='MSELoss')

        x1 = nn.ReLU()(x1-K)  # @ NT1: hand in min(K,xt1)
        x2 = nn.ReLU()(x2-K)
    # @ NT2 --> Loss for Prd2: loss_y
    loss_y1=Loss(pred=y1/w,targ=target_Y(x_t2=x1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type=loss_type)
    loss_y2=Loss(pred=y2/w,targ=target_Y(x_t2=x2,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type=loss_type)
    
    loss=(loss_v1+loss_u1+loss_y1)+(loss_v2+loss_u2+loss_y2)
    return loss.to(device)
    
def get_target_path(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, main_models}
  '''
  Perform the optimized stepwise approximation with inference_mode on. 
  '''
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
  dB1=pop1_dict['dB']
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
  dB2=pop2_dict['dB']
  ## -------------------------------- Common Params -------------------------------- ##
  w=pop1_dict['GlobalParams'].w
  K=pop1_dict['GlobalParams'].K
  dt=pop1_dict['GlobalParams'].dt
  NT1=pop1_dict['GlobalParams'].NT1
  NT2=pop1_dict['GlobalParams'].NT2
  NumTrain=pop1_dict['GlobalParams'].NumTrain
  device=pop1_dict['GlobalParams'].device
  trick=pop1_dict['GlobalParams'].trick
  omiga1=(pi1/gamma1)/(pi1/gamma1+pi2/gamma2)
  omiga2=(pi2/gamma2)/(pi1/gamma1+pi2/gamma2)
  ## -------------------------------- Paths -------------------------------- ##
  x1_path = torch.ones(NumTrain,NT2+1).to(device)
  x2_path = torch.ones(NumTrain,NT2+1).to(device)

  v1_path = torch.ones(NumTrain,NT2+1).to(device)
  v2_path = torch.ones(NumTrain,NT2+1).to(device)

  u1_path = torch.ones(NumTrain,NT2+1).to(device)
  u2_path = torch.ones(NumTrain,NT2+1).to(device)

  y1_path = torch.ones(NumTrain,NT2+1).to(device)
  y2_path = torch.ones(NumTrain,NT2+1).to(device)

  g1_path = torch.ones(NumTrain,NT2+1).to(device)
  g2_path = torch.ones(NumTrain,NT2+1).to(device)
  
  Gamma1_path = torch.ones(NumTrain,NT2+1).to(device)
  Gamma2_path = torch.ones(NumTrain,NT2+1).to(device)
  
  a1_path = torch.ones(NumTrain,NT2+1).to(device)
  a2_path = torch.ones(NumTrain,NT2+1).to(device)

  S_path=torch.ones(NT2+1).to(device)

  with torch.inference_mode():
    if trick=='logit': ## use yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde), dyx=-zx*(1+yx)*yx*dB
      for j in range(0, NT2+1):
        if j==0:
          v1_tilde=w*main_models1.v0_model(x1).view(-1,1)
          v2_tilde=w*main_models2.v0_model(x2).view(-1,1)
          v1=w*torch.sigmoid(v1_tilde/w).view(-1,1).to(device)
          v2=w*torch.sigmoid(v2_tilde/w).view(-1,1).to(device)

          u1_tilde=w*main_models1.u0_model(x1).view(-1,1)
          u2_tilde=w*main_models2.u0_model(x2).view(-1,1)
          u1=w*torch.sigmoid(u1_tilde/w).view(-1,1).to(device)
          u2=w*torch.sigmoid(u2_tilde/w).view(-1,1).to(device)

          y1_tilde=w*main_models1.y0_model(x1).view(-1,1)
          y2_tilde=w*main_models2.y0_model(x2).view(-1,1)
          y1=w*torch.sigmoid(y1_tilde/w).view(-1,1).to(device)
          y2=w*torch.sigmoid(y2_tilde/w).view(-1,1).to(device)
          
          S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
        else:
          x1 =x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
          x2 =x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

          c1=c1+a1*dt
          c2=c2+a2*dt

          if j<=NT1:
            zv1 = main_models1.zv_models[j-1](x1)
            zv2 = main_models2.zv_models[j-1](x2)

            zu1 = main_models1.zu_models[j-1](x1)
            zu2 = main_models2.zu_models[j-1](x2)
            
          zy1 = main_models1.zy_models[j-1](x1)
          zy2 = main_models2.zy_models[j-1](x2)
          
          v1_tilde=v1_tilde+(zv1**2)*(v1-w/2)*dt+w*zv1*dB1[:,j].view(-1,1) if j<=NT1 else v1_tilde
          v2_tilde=v2_tilde+(zv2**2)*(v2-w/2)*dt+w*zv2*dB2[:,j].view(-1,1) if j<=NT1 else v2_tilde
          v1=w*torch.sigmoid(v1_tilde/w).view(-1,1).to(device) if j<=NT1 else v1
          v2=w*torch.sigmoid(v2_tilde/w).view(-1,1).to(device) if j<=NT1 else v2

          u1_tilde=u1_tilde+(zu1**2)*(u1-w/2)*dt+w*zu1*dB1[:,j].view(-1,1) if j<=NT1 else u1
          u2_tilde=u2_tilde+(zu2**2)*(u2-w/2)*dt+w*zu2*dB2[:,j].view(-1,1) if j<=NT1 else u2
          u1=w*torch.sigmoid(u1_tilde/w).view(-1,1).to(device) if j<=NT1 else u1
          u2=w*torch.sigmoid(u2_tilde/w).view(-1,1).to(device) if j<=NT1 else u2

          y1_tilde=y1_tilde+(zy1**2)*(y1-w/2)*dt+w*zy1*dB1[:,j].view(-1,1)
          y2_tilde=y2_tilde+(zy2**2)*(y2-w/2)*dt+w*zy2*dB2[:,j].view(-1,1)
          y1=(w*torch.sigmoid(y1_tilde/w).view(-1,1)).to(device)
          y2=(w*torch.sigmoid(y2_tilde/w).view(-1,1)).to(device)

          S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
        
        g1=(v1+u1)/zeta1 if j<=NT1 else y1/zeta1
        g2=(v2+u2)/zeta2 if j<=NT1 else y2/zeta2

        Gamma1=(v1+u1-S)/gamma1 if j<=NT1 else (y1-S)/gamma1
        Gamma2=(v2+u2-S)/gamma2 if j<=NT1 else (y2-S)/gamma2

        a1=((NT1-j)*(v1+u1)+(NT2-NT1)*y1)*dt/beta1 if j<=NT1 else (NT2-j)*y1*dt/beta1
        a2=((NT1-j)*(v2+u2)+(NT2-NT1)*y2)*dt/beta2 if j<=NT1 else (NT2-j)*y2*dt/beta2
        ## -------------------------------- Record Paths -------------------------------- ##
        x1_path[:,j] = x1.squeeze()
        x2_path[:,j] = x2.squeeze()
        
        v1_path[:,j] = v1.squeeze()
        v2_path[:,j] = v2.squeeze()

        u1_path[:,j] = u1.squeeze()
        u2_path[:,j] = u2.squeeze()

        y1_path[:,j] = y1.squeeze()
        y2_path[:,j] = y2.squeeze()

        g1_path[:,j] = g1.squeeze()
        g2_path[:,j] = g2.squeeze()

        Gamma1_path[:,j] = Gamma1.squeeze()
        Gamma2_path[:,j] = Gamma2.squeeze()

        a1_path[:,j] = a1.squeeze()
        a2_path[:,j] = a2.squeeze()

        S_path[j]=S
        ## -------------------------------- ------------ -------------------------------- ##
        if j==NT1:  # @ NT1 --> hand in min(K,xt1); init for Prd2 (with remaining inventory)
          x1 = nn.ReLU()(x1-K)   
          x2 = nn.ReLU()(x2-K)

    if trick=='clamp': ## use dyx=-zx*(1+yx)*yx*dB ONLY
      for j in range(0, NT2+1):
        if j==0:
          v1=w*main_models1.v0_model(x1).view(-1,1)#.clamp(min=0,max=1)
          v2=w*main_models2.v0_model(x2).view(-1,1)#.clamp(min=0,max=1)
          # print(j,x1v1[:5])
          u1=w*main_models1.u0_model(x1).view(-1,1)#.clamp(min=0,max=1)
          u2=w*main_models2.u0_model(x2).view(-1,1)#.clamp(min=0,max=1)
          
          y1=w*main_models1.y0_model(x1).view(-1,1)#.clamp(min=0,max=1)
          y2=w*main_models2.y0_model(x2).view(-1,1)#.clamp(min=0,max=1)
          
          S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
        else:   ### j>0:
          x1 =x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
          x2 =x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

          c1=c1+a1*dt
          c2=c2+a2*dt

          if j<=NT1:
            zv1 = main_models1.zv_models[j-1](x1)
            zv2 = main_models2.zv_models[j-1](x2)

            zu1 = main_models1.zu_models[j-1](x1)
            zu2 = main_models2.zu_models[j-1](x2)
            
          zy1 = main_models1.zy_models[j-1](x1)
          zy2 = main_models2.zy_models[j-1](x2)
          
          v1=((v1+v1*(1-v1/w)*zv1*dB1[:,j].view(-1,1)) if j<=NT1 else v1).clamp(min=0,max=w)
          v2=((v2+v2*(1-v2/w)*zv2*dB2[:,j].view(-1,1)) if j<=NT1 else v2).clamp(min=0,max=w)

          u1=((u1+u1*(1-u1/w)*zu1*dB1[:,j].view(-1,1)) if j<=NT1 else u1).clamp(min=0,max=w)
          u2=((u2+u2*(1-u2/w)*zu2*dB2[:,j].view(-1,1)) if j<=NT1 else u2).clamp(min=0,max=w)

          y1=((y1+y1*(1-y1/w)*zy1*dB1[:,j].view(-1,1))).clamp(min=0,max=w)
          y2=((y2+y2*(1-y2/w)*zy2*dB2[:,j].view(-1,1))).clamp(min=0,max=w)

          S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
        
        g1=(v1+u1)/zeta1 if j<=NT1 else y1/zeta1
        g2=(v2+u2)/zeta2 if j<=NT1 else y2/zeta2

        Gamma1=(v1+u1-S)/gamma1 if j<=NT1 else (y1-S)/gamma1
        Gamma2=(v2+u2-S)/gamma2 if j<=NT1 else (y2-S)/gamma2

        a1=((NT1-j)*(v1+u1)+(NT2-NT1)*y1)*dt/beta1 if j<=NT1 else (NT2-j)*y1*dt/beta1
        a2=((NT1-j)*(v2+u2)+(NT2-NT1)*y2)*dt/beta2 if j<=NT1 else (NT2-j)*y2*dt/beta2
        ## -------------------------------- Record Paths -------------------------------- ##
        x1_path[:,j] = x1.squeeze()
        x2_path[:,j] = x2.squeeze()
        
        v1_path[:,j] = v1.squeeze()
        v2_path[:,j] = v2.squeeze()

        u1_path[:,j] = u1.squeeze()
        u2_path[:,j] = u2.squeeze()

        y1_path[:,j] = y1.squeeze()
        y2_path[:,j] = y2.squeeze()

        g1_path[:,j] = g1.squeeze()
        g2_path[:,j] = g2.squeeze()

        Gamma1_path[:,j] = Gamma1.squeeze()
        Gamma2_path[:,j] = Gamma2.squeeze()

        a1_path[:,j] = a1.squeeze()
        a2_path[:,j] = a2.squeeze()

        S_path[j]=S
        ## -------------------------------- ------------ -------------------------------- ##
        if j==NT1:  # @ NT1 --> hand in min(K,xt1); init for Prd2 (with remaining inventory)
          x1 = nn.ReLU()(x1-K)   
          x2 = nn.ReLU()(x2-K)

  cum_g1_path=torch.zeros(size=(NumTrain,1)).to(device)
  cum_g1_path=torch.hstack([cum_g1_path,dt*(g1_path.cumsum(axis=1)[:,:-1])])
  cum_g2_path=torch.zeros(size=(NumTrain,1)).to(device)
  cum_g2_path=torch.hstack([cum_g2_path,dt*(g2_path.cumsum(axis=1)[:,:-1])])

  cum_Gamma1_path=torch.zeros(size=(NumTrain,1)).to(device)
  cum_Gamma1_path=torch.hstack([cum_Gamma1_path,dt*(Gamma1_path.cumsum(axis=1)[:,:-1])])
  cum_Gamma2_path=torch.zeros(size=(NumTrain,1)).to(device)
  cum_Gamma2_path=torch.hstack([cum_Gamma2_path,dt*(Gamma2_path.cumsum(axis=1)[:,:-1])])
  
  cum_a1_path=torch.zeros(size=(NumTrain,1)).to(device)
  cum_a1_path=torch.hstack([cum_a1_path,dt*(a1_path.cumsum(axis=1)[:,:-1])])
  cum_a2_path=torch.zeros(size=(NumTrain,1)).to(device)
  cum_a2_path=torch.hstack([cum_a2_path,dt*(a2_path.cumsum(axis=1)[:,:-1])])
  
  base1_path=torch.FloatTensor([h1]*(NT2+1)).to(device)
  base2_path=torch.FloatTensor([h2]*(NT2+1)).to(device)
  cum_base1_path=torch.linspace(0,h1,NT2+1).to(device)
  cum_base2_path=torch.linspace(0,h2,NT2+1).to(device)

  pop1_path_dict={'inventory':x1_path,
                  'price':S_path,
                  'expansion':a1_path,
                  'generation':g1_path,
                  'trading':Gamma1_path,
                  'base':base1_path,
                  'cum_expansion':cum_a1_path,
                  'cum_generation':cum_g1_path,
                  'cum_trading':cum_Gamma1_path,
                  'cum_base':cum_base1_path,
                  'v':v1_path,
                  'u':u1_path,
                  'y':y1_path}
  pop2_path_dict={'inventory':x2_path,
                  'price':S_path,
                  'expansion':a2_path,
                  'generation':g2_path,
                  'trading':Gamma2_path,
                  'base':base2_path,
                  'cum_expansion':cum_a2_path,
                  'cum_generation':cum_g2_path,
                  'cum_trading':cum_Gamma2_path,
                  'cum_base':cum_base2_path,
                  'v':v2_path,
                  'u':u2_path,
                  'y':y2_path}
  
  return pop1_path_dict, pop2_path_dict

class plot_results():
    '''
    A class of assembled functions to evaluate and visualize training results and model performances. 
    '''
    def __init__(self,pop1_dict, pop2_dict, loss, PlotPaths=100, seed=42): #dB, init_x, init_c, GlobalParams, main_models, loss,PlotPaths=100, seed=42):
        ## -------------------------------- Common Params -------------------------------- ##
        self.loss=loss
        self.target_type='indicator' if (pop1_dict['GlobalParams'].target_type=='indicator' and pop1_dict['GlobalParams'].trick!='logit') else "sigmoid"
        self.delta=pop1_dict['GlobalParams'].delta
        self.K=pop1_dict['GlobalParams'].K
        self.w=pop1_dict['GlobalParams'].w
        self.NT1=pop1_dict['GlobalParams'].NT1
        self.NT2=pop1_dict['GlobalParams'].NT2
        self.dt=pop1_dict['GlobalParams'].dt
        self.t = np.array([i for i in range(self.NT2+1)]) * self.dt
        self.NumTrain=pop1_dict['GlobalParams'].NumTrain
        self.number_of_paths=np.minimum(PlotPaths,self.NumTrain)
        self.seed=seed
        ## -------------------------------- P1 Params -------------------------------- ##
        self.GlobalParams1=pop1_dict['GlobalParams']
        self.h1=pop1_dict['GlobalParams'].h
        ## -------------------------------- P2 Params -------------------------------- ##
        self.GlobalParams2=pop2_dict['GlobalParams']
        self.h2=pop2_dict['GlobalParams'].h

        ## -------------------------------- Process Paths -------------------------------- ##
        self.pop1_path_dict,self.pop2_path_dict=get_target_path(pop1_dict, pop2_dict)
        torch.manual_seed(self.seed)
        idx_list = np.random.choice(self.NumTrain, self.number_of_paths, replace = False)
        self.pop1_plot = {k:v.detach().numpy()[idx_list] for k,v in self.pop1_path_dict.items() if v.shape[0]==self.NumTrain}
        self.pop2_plot = {k:v.detach().numpy()[idx_list] for k,v in self.pop2_path_dict.items() if v.shape[0]==self.NumTrain}
        self.pop1_plot.update({k:v.detach().numpy() for k,v in self.pop1_path_dict.items() if v.shape[0]!=self.NumTrain})
        self.pop2_plot.update({k:v.detach().numpy() for k,v in self.pop2_path_dict.items() if v.shape[0]!=self.NumTrain})
    
    def FwdLoss(self,log=True):
        plt.figure(figsize=(8,5))
        plt.title("Forward_Loss vs Epoch",fontsize=18)
        plt.plot(self.loss,color='firebrick',alpha=0.6)
        if log==True:
            plt.yscale('log')

    def Inventory_And_Price(self,Histogram=True): #,single_sample=False):
        plt.figure(figsize=(14,6))
        plt.subplot(121)
        plt.title("$Inventory(X_t)$")
        for i in range(self.number_of_paths):
          ax1=plt.plot(self.t[:self.NT1+1],self.pop1_plot['inventory'][i][:self.NT1+1], self.t[self.NT1+1:],self.pop1_plot['inventory'][i][self.NT1+1:],color="green", alpha=0.3)
          ax2=plt.plot(self.t[:self.NT1+1],self.pop2_plot['inventory'][i][:self.NT1+1], self.t[self.NT1+1:],self.pop2_plot['inventory'][i][self.NT1+1:],color="firebrick", alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(122)
        plt.title("$Price(S_t)$")
        plt.plot(self.t, self.pop1_plot['price'],color='darkgrey')
        plt.ylim(-0.1,1.5)
        plt.show()

        if Histogram==True:
          x1_t1=self.pop1_path_dict['inventory'][:,self.NT1]
          x2_t1=self.pop2_path_dict['inventory'][:,self.NT1]
          x1_t2=self.pop1_path_dict['inventory'][:,self.NT2]
          x2_t2=self.pop2_path_dict['inventory'][:,self.NT2]
          ## -------------------------------- Population 1 -------------------------------- ##
          plt.figure(figsize=(12,5))
          plt.suptitle("Inventory @ Delivery - P1")

          plt.subplot(121)
          plt.title("Distribution of $X_{T_1}^{(1)}$")
          sns.histplot(data=x1_t1, bins=100,stat='count',alpha=0.6,color= 'green')
          sns.kdeplot(x1_t1, color="green",label='P1')
          plt.xlabel("$X_{T_1}^{(1)}$")
          plt.ylabel("Count")
          plt.legend()

          plt.subplot(122)
          plt.title("Distribution of $X_{T_2}^{(1)}$")
          sns.histplot(data=x1_t2, bins=100,stat='count',alpha=0.6,color= 'green')
          sns.kdeplot(x1_t2, color="green",label='P1')
          plt.xlabel("$X_{T_2}^{(1)}$")
          plt.ylabel("Count")
          plt.legend()
          ## -------------------------------- Population 2 -------------------------------- ##
          plt.figure(figsize=(12,5))
          plt.suptitle("Inventory @ Delivery - P2")

          plt.subplot(121)
          plt.title("Distribution of $X_{T_1}^{(2)}$")
          sns.histplot(data=x2_t1, bins=100,stat='count',alpha=0.6,color= 'firebrick')
          sns.kdeplot(x2_t1, color="firebrick",label='P2')
          plt.xlabel("$X_{T_1}^{(2)}$")
          plt.ylabel("Count")
          plt.legend()

          plt.subplot(122)
          plt.title("Distribution of $X_{T_2}^{(2)}$")
          sns.histplot(data=x2_t2, bins=100,stat='count',alpha=0.6,color= 'firebrick')
          sns.kdeplot(x2_t2, color="firebrick",label='P2')
          plt.xlabel("$X_{T_2}^{(2)}$")
          plt.ylabel("Count")
          plt.legend()

          plt.show()


    def Decomposition_Inventory(self, cumulative=True, base_rate=False):
        ## [0,NT1] --> init NT1 --> [NT1+1,NT2]
        plt.figure(figsize=(15,4) if base_rate==False else (18,4))
        if base_rate==True:
          plt.subplot(141)
          plt.title("Base Rate")
          ax1=plt.plot(self.t,self.pop1_plot['base'], color="green")
          ax2=plt.plot(self.t,self.pop2_plot['base'], color="firebrick")
          # plt.ylim(-0.5,1) 
          plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(131 if base_rate==False else 142)
        plt.title("Expansion Rate")
        for i in range(self.number_of_paths):
          ax1=plt.plot(self.t,self.pop1_plot['expansion'][i], color="green", alpha=0.3)
          ax2=plt.plot(self.t,self.pop2_plot['expansion'][i], color="firebrick", alpha=0.3)
        # plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(132 if base_rate==False else 143)
        plt.title("Generation Rate")
        for i in range(self.number_of_paths):
          ax1=plt.plot(self.t,self.pop1_plot['generation'][i], color="green", alpha=0.3)
          ax2=plt.plot(self.t,self.pop2_plot['generation'][i], color="firebrick", alpha=0.3)
        # plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(133 if base_rate==False else 144)
        plt.title("Trading Rate")
        for i in range(self.number_of_paths):
          ax1=plt.plot(self.t,self.pop1_plot['trading'][i], color="green", alpha=0.3)
          ax2=plt.plot(self.t,self.pop2_plot['trading'][i], color="firebrick", alpha=0.3)
        # plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        ## Accumulated Inventory - Decomposition
        if cumulative==True:
            plt.figure(figsize=(15,4) if base_rate==False else (18,4))
            if base_rate==True:
                plt.subplot(141)
                plt.title("Accumulated Baseline Growth")
                ax1=plt.plot(self.t,self.pop1_plot['cum_base'], color="green")
                ax2=plt.plot(self.t,self.pop2_plot['cum_base'], color="firebrick")
                plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(131 if base_rate==False else 142)
            plt.title("Accumulated Expansion")  ## TODO: change to "Accumulated Baseline Rate" by starting from baseline rates instead of 0.
            for i in range(self.number_of_paths):
              ax1=plt.plot(self.t,self.pop1_plot['cum_expansion'][i], color="green", alpha=0.3)
              ax2=plt.plot(self.t,self.pop2_plot['cum_expansion'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(132 if base_rate==False else 143)
            plt.title("Accumulated Generation")
            for i in range(self.number_of_paths):
              ax1=plt.plot(self.t,self.pop1_plot['cum_generation'][i], color="green", alpha=0.3)
              ax2=plt.plot(self.t,self.pop2_plot['cum_generation'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(133 if base_rate==False else 144)
            plt.title("Accumulated Trading")
            for i in range(self.number_of_paths):
              ax1=plt.plot(self.t,self.pop1_plot['cum_trading'][i], color="green", alpha=0.3)
              ax2=plt.plot(self.t,self.pop2_plot['cum_trading'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})

    def Key_Processes(self,V=True, U=True, Y=True):
        num_plots=int(V+U+Y)
        p=1
        plt.figure(figsize=(num_plots*5,4))
        if V==True:
            plt.subplot(int(f'1{num_plots}{p}'))
            p+=1
            plt.title("$V_t$")
            for i in range(self.number_of_paths):
                # print(self.t.shape,self.pop1_path_dict['v'][i].shape)
                ax1=plt.plot(self.t,self.pop1_path_dict['v'][i], color="green", alpha=0.3)
                ax2=plt.plot(self.t,self.pop2_path_dict['v'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})
        if U==True:
            plt.subplot(int(f'1{num_plots}{p}'))
            p+=1
            plt.title("$U_t$")
            for i in range(self.number_of_paths):
                ax3=plt.plot(self.t, self.pop1_path_dict['u'][i], color="green", alpha=0.3)
                ax4=plt.plot(self.t, self.pop2_path_dict['u'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax3,'P2':ax4})
        if Y==True:
            plt.subplot(int(f'1{num_plots}{p}'))
            p+=1
            plt.title("$Y_t$")
            for i in range(self.number_of_paths):
                ax5=plt.plot(self.t, self.pop1_path_dict['y'][i], color="green", alpha=0.3)
                ax6=plt.plot(self.t, self.pop2_path_dict['y'][i], color="firebrick", alpha=0.3)
            plt.legend({'P1':ax5,'P2':ax6})

    def Terminal_Convergence(self,QQ_plot=False,Fitted=True):
        str_x1_t1, str_x1_t2 = ("$X^{(1)}_{T_1}$", "$X^{(1)}_{T_2}$")
        str_v1_t1, str_v1_t2 = ("$V^{(1)}_{T_1}$", "$V^{(1)}_{T_2}$")
        str_u1_t1, str_u1_t2 = ("$U^{(1)}_{T_1}$", "$U^{(1)}_{T_2}$")
        str_y1_t1, str_y1_t2 = ("$Y^{(1)}_{T_1}$", "$Y^{(1)}_{T_2}$")
        
        str_x2_t1, str_x2_t2 = ("$X^{(2)}_{T_1}$", "$X^{(2)}_{T_2}$")
        str_v2_t1, str_v2_t2 = ("$V^{(2)}_{T_1}$", "$V^{(2)}_{T_2}$")
        str_u2_t1, str_u2_t2 = ("$U^{(2)}_{T_1}$", "$U^{(2)}_{T_2}$")
        str_y2_t1, str_y2_t2 = ("$Y^{(2)}_{T_1}$", "$Y^{(2)}_{T_2}$")

        str_u_t1 , str_v_t1 , str_y_t1 , str_y_t2, str_x_t1, str_x_t2 = ("$U_{T_1}$", "$V_{T_1}$", "$Y_{T_1}$", "$Y_{T_2}$", "$X_{T_1}$","$X_{T_2}$")
        ## -------------------------------- Targets -------------------------------- ##
        x1_t1=self.pop1_path_dict['inventory'][:,self.NT1]
        x2_t1=self.pop2_path_dict['inventory'][:,self.NT1]  ##[0,NT1] --> init @ NT1 --> [NT1,Nt2]
        target_v1_t1=self.w*target_V(x_t1=x1_t1,GlobalParams=self.GlobalParams1,target_type=self.target_type).detach().numpy()
        target_v2_t1=self.w*target_V(x_t1=x2_t1,GlobalParams=self.GlobalParams2,target_type=self.target_type).detach().numpy()
        target_u1_t1=self.w*target_U(x_t1=x1_t1,y_t1=self.pop1_path_dict['y'][:,self.NT1]/self.w,GlobalParams=self.GlobalParams1,target_type=self.target_type).detach().numpy()
        target_u2_t1=self.w*target_U(x_t1=x2_t1,y_t1=self.pop2_path_dict['y'][:,self.NT1]/self.w,GlobalParams=self.GlobalParams2,target_type=self.target_type).detach().numpy()
          
        x1_t2=self.pop1_path_dict['inventory'][:,self.NT2]
        x2_t2=self.pop2_path_dict['inventory'][:,self.NT2]
        target_y1_t2=self.w*target_Y(x_t2=x1_t2,GlobalParams=self.GlobalParams1,target_type=self.target_type).detach().numpy()
        target_y2_t2=self.w*target_Y(x_t2=x2_t2,GlobalParams=self.GlobalParams2,target_type=self.target_type).detach().numpy()

        if QQ_plot==True:
          plt.figure(figsize=(15,4))

          plt.subplot(131)
          plt.title(f"QQ-Plot of {str_v1_t1} , {str_v2_t1}")
          ax1=plt.scatter(target_v1_t1,self.pop1_path_dict['v'][:,-1],s=3,alpha=0.3,color='green')
          ax2=plt.scatter(target_v2_t1,self.pop2_path_dict['v'][:,-1],s=3,alpha=0.3,color='firebrick')
          plt.plot(np.linspace(-1,1,5),np.linspace(-1,1,5),linestyle='--',linewidth=1,color='r')
          plt.legend({'P1':ax1,'P2':ax2})
          # plt.xlim(-0.1,1.1)
          # plt.ylim(-0.1,1.1)
          plt.xlabel("Target")
          plt.ylabel("$V_{T_1}$")

          plt.subplot(132)
          plt.title(f"QQ-Plot of {str_u1_t1} , {str_u2_t1}")
          ax1=plt.scatter(target_u1_t1 ,self.pop1_path_dict['u'][:,-1],s=3,alpha=0.3,color='green')
          ax2=plt.scatter(target_u2_t1,self.pop2_path_dict['u'][:,-1],s=3,alpha=0.3,color='firebrick')
          plt.plot(np.linspace(-1,1,5),np.linspace(-1,1,5),linestyle='--',linewidth=1,color='r')
          plt.legend({'P1':ax1,'P2':ax2})
          # plt.xlim(-0.1,1.1)
          # plt.ylim(-0.1,1.1)
          plt.xlabel("Target")
          plt.ylabel("$U_{T_1}$")

          plt.subplot(133)
          plt.title(f"QQ-Plot of {str_y1_t2} , {str_y2_t2}")
          ax1=plt.scatter(target_y1_t2 ,self.pop1_path_dict['y'][:,-1],s=3,alpha=0.3,color='green')
          ax2=plt.scatter(target_y2_t2,self.pop2_path_dict['y'][:,-1],s=3,alpha=0.3,color='firebrick')
          plt.plot(np.linspace(-1,1,5),np.linspace(-1,1,5),linestyle='--',linewidth=1,color='r')
          plt.legend({'P1':ax1,'P2':ax2})
          # plt.xlim(-0.1,1.1)
          # plt.ylim(-0.1,1.1)
          plt.xlabel("Target")
          plt.ylabel("$Y_{T_2}$")
 
        if Fitted==True:
          ## -------------------------------- Population 1 -------------------------------- ##
          plt.figure(figsize=(15,13))
          plt.suptitle(f"Termianl Values")

          plt.subplot(231)
          plt.title(str_v1_t1)
          ax1=plt.scatter(x1_t1,target_v1_t1,s=3,alpha=0.3,color='black')
          ax2=plt.scatter(x1_t1,self.pop1_path_dict['v'][:,self.NT1],s=3,alpha=0.5,color='green')
          # legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x1_t1}<0.9)':ax1,f'Learnt: {str_v1_t1}':ax2},loc='upper right', fontsize=8,framealpha=0.6)#, bbox_to_anchor=(0.5, -0.1))
          plt.xlabel(str_x1_t1)
          plt.ylabel(str_v1_t1)
          # for handle, text in zip(legend.legend_handles, legend.get_texts()):
          #   handle.set_alpha(1)  
          #   text.set_color(handle.get_facecolor()[0]) 
          # plt.tight_layout(rect=[0, 0.1, 1, 1])  


          plt.subplot(232)
          plt.title(str_u1_t1)
          ax1=plt.scatter(x1_t1,target_u1_t1,s=3,alpha=0.2,color='black')
          ax2=plt.scatter(x1_t1,self.pop1_path_dict['u'][:,self.NT1],s=3,alpha=0.5,color='green')
          # legend=plt.legend({f'Target: {self.target_type}({str_x1_t1}>0.9)*{str_y1_t1}':ax1,f'Learnt: {str_u1_t1}':ax2},loc='upper right', fontsize=8,framealpha=0.6)#, bbox_to_anchor=(0.5, -0.1))
          plt.xlabel(str_x1_t1)
          plt.ylabel(str_u1_t1)
          # for handle, text in zip(legend.legend_handles, legend.get_texts()):
          #   handle.set_alpha(1)  
          #   text.set_color(handle.get_facecolor()[0]) 
          

          plt.subplot(233)
          plt.title(str_y1_t2)
          ax1=plt.scatter(x1_t2,target_y1_t2,s=3,alpha=0.2,color='black')
          ax2=plt.scatter(x1_t2,self.pop1_path_dict['y'][:,self.NT2],s=3,alpha=0.5,color='green')
          # legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x1_t2}<0.9)':ax1,f'Learnt: {str_y1_t2}':ax2},loc='upper right', fontsize=8,framealpha=0.6)#, bbox_to_anchor=(0.75, -0.1))
          plt.xlabel(str_x1_t2)
          plt.ylabel(str_y1_t2)
          # for handle, text in zip(legend.legend_handles, legend.get_texts()):
          #   handle.set_alpha(1)  
          #   text.set_color(handle.get_facecolor()[0]) 
          ## -------------------------------- Population 2 -------------------------------- ##
          plt.subplot(234)
          plt.title(str_v2_t1)
          ax1=plt.scatter(x2_t1,target_v2_t1,s=3,alpha=0.3,color='black')
          ax2=plt.scatter(x2_t1,self.pop2_path_dict['v'][:,self.NT1],s=3,alpha=0.5,color='green')
          plt.xlabel(str_x2_t1)
          plt.ylabel(str_v2_t1)
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t1}<0.9)':ax1,f'Learnt: {str_v_t1}':ax2}, fontsize=9,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 

          plt.subplot(235)
          plt.title(str_u2_t1)
          ax1=plt.scatter(x2_t1,target_u2_t1,s=3,alpha=0.2,color='black')
          ax2=plt.scatter(x2_t1,self.pop2_path_dict['u'][:,self.NT1],s=3,alpha=0.5,color='green')
          plt.xlabel(str_x2_t1)
          plt.ylabel(str_u2_t1)
          legend=plt.legend({f'Target: {self.target_type}({str_x_t1}>0.9)*{str_y_t1}':ax1,f'Learnt: {str_u_t1}':ax2}, fontsize=9,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 

          plt.subplot(236)
          plt.title(str_y2_t2)
          ax1=plt.scatter(x2_t2,target_y2_t2,s=3,alpha=0.2,color='black')
          ax2=plt.scatter(x2_t2,self.pop2_path_dict['y'][:,self.NT2],s=3,alpha=0.5,color='green')
          plt.xlabel(str_x2_t2)
          plt.ylabel(str_y2_t2)
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t2}<0.9)':ax1,f'Learnt: {str_y_t2}':ax2},fontsize=9,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 

          plt.tight_layout(rect=[0, 0.25, 1, 1])  
          plt.subplots_adjust(hspace=0.26)
        # if Histogram==True: 
        #   idx0_v1=np.argwhere(target_v1_t1<0.5).reshape(-1,)  # VT1==0
        #   idx0_v2=np.argwhere(target_v2_t1<0.5).reshape(-1,)  # VT1==0
        #   idx1_v1=np.argwhere(target_v1_t1>0.5).reshape(-1,)  # VT1==1
        #   idx1_v2=np.argwhere(target_v1_t1>0.5).reshape(-1,)  # VT1==1
          
        #   idx0_y1=np.argwhere(target_Y(x_t1=self.pop1_path_dict['inventory'][:,self.NT1],x_t2=self.pop1_path_dict['inventory'][:,self.NT2],GlobalParams=self.GlobalParams1,target_type='indicator').detach().numpy()==0).reshape(-1,)  # YT2==0
        #   idx0_y2=np.argwhere(target_Y(x_t1=self.pop2_path_dict['inventory'][:,self.NT1],x_t2=self.pop2_path_dict['inventory'][:,self.NT2],GlobalParams=self.GlobalParams2,target_type='indicator').detach().numpy()==0).reshape(-1,)  # YT2==0
        #   idx1_y1=np.argwhere(target_Y(x_t1=self.pop1_path_dict['inventory'][:,self.NT1],x_t2=self.pop1_path_dict['inventory'][:,self.NT2],GlobalParams=self.GlobalParams1,target_type='indicator').detach().numpy()==1).reshape(-1,)  # YT2==0
        #   idx1_y2=np.argwhere(target_Y(x_t1=self.pop2_path_dict['inventory'][:,self.NT1],x_t2=self.pop2_path_dict['inventory'][:,self.NT2],GlobalParams=self.GlobalParams2,target_type='indicator').detach().numpy()==1).reshape(-1,)  # YT2==0
        #   ## -------------------------------- Population 1 -------------------------------- ##
        #   plt.figure(figsize=(20,6))
        #   plt.suptitle("Convergency - P1")

        #   plt.subplot(131)
        #   plt.title("Distribution of $V_{T_1}^{(1)}$")
        #   sns.histplot(data=self.pop1_path_dict['v'][idx0_v1,self.NT1], bins=100,stat='count',alpha=0.6,color= 'green')
        #   sns.histplot(data=self.pop1_path_dict['v'][idx1_v1,self.NT1], bins=100,stat='count',alpha=0.3,color= 'grey')
        #   sns.kdeplot(self.pop1_path_dict['v'][idx0_v1,self.NT1], color="green",label=('$X_{T_1}^{(1)}$>'+f'{self.K}, '+'$V_{T_1}^{(1)}\\rightarrow$ 0'))
        #   sns.kdeplot(self.pop1_path_dict['v'][idx1_v1,self.NT1], color="black",label=('$X_{T_1}^{(1)}$<'+f'{self.K}, '+'$V_{T_1}^{(1)}\\rightarrow$ 1'))
        #   plt.xlabel("$V_{T_1}^{(1)}$")
        #   plt.ylabel("$V_{T_1}^{(1)}$ Count")
        #   plt.legend()

        #   plt.subplot(132)
        #   plt.title("Distribution of $U_{T_1}^{(1)}$")
        #   sns.histplot(data=self.pop1_path_dict['u'][:,self.NT1], bins=100,stat='count',alpha=0.6,color= 'green')
        #   sns.kdeplot(self.pop1_path_dict['u'][:,self.NT1], color="green")
        #   plt.xlabel("$U_{T_1}^{(1)}$")
        #   plt.ylabel("$U_{T_1}^{(1)}$ Count")

        #   plt.subplot(133)
        #   plt.title("Distribution of $Y_{T_2}^{(1)}$")
        #   sns.histplot(data=self.pop1_path_dict['y'][idx0_y1,self.NT2], bins=100,stat='count',alpha=0.6,color= 'green')
        #   sns.histplot(data=self.pop1_path_dict['y'][idx1_y1,self.NT2], bins=100,stat='count',alpha=0.3,color= 'grey')
        #   sns.kdeplot(self.pop1_path_dict['y'][idx0_y1,self.NT2], color="green",label=('$Y_{T_2}^{(1)}\\rightarrow$ 0'))
        #   sns.kdeplot(self.pop1_path_dict['y'][idx1_y1,self.NT2], color="black",label=('$Y_{T_2}^{(1)}\\rightarrow$ 1'))
        #   plt.xlabel("$Y_{T_2}^{(1)}$")
        #   plt.ylabel("$Y_{T_2}^{(1)}$ Count")
        #   plt.legend()
        #   ## -------------------------------- Population 2 -------------------------------- ##
        #   plt.figure(figsize=(20,6))
        #   plt.suptitle("Convergency - P2")

        #   plt.subplot(131)
        #   plt.title("Distribution of $V_{T_1}^{(2)}$")
        #   sns.histplot(data=self.pop2_path_dict['v'][idx0_v2,self.NT1], bins=100,stat='count',alpha=0.6,color= 'green')
        #   sns.histplot(data=self.pop2_path_dict['v'][idx1_v2,self.NT1], bins=100,stat='count',alpha=0.3,color= 'grey')
        #   sns.kdeplot(self.pop2_path_dict['v'][idx0_v2,self.NT1], color="green",label=('$X_{T_1}^{(2)}$>'+f'{self.K}, '+'$V_{T_1}^{(2)}\\rightarrow$ 0'))
        #   sns.kdeplot(self.pop2_path_dict['v'][idx1_v2,self.NT1], color="black",label=('$X_{T_1}^{(2)}$<'+f'{self.K}, '+'$V_{T_1}^{(2)}\\rightarrow$ 1'))
        #   plt.xlabel("$V_{T_1}^{(2)}$")
        #   plt.ylabel("$V_{T_1}^{(2)}$ Count")
        #   plt.legend()

        #   plt.subplot(132)
        #   plt.title("Distribution of $U_{T_1}^{(2)}$")
        #   sns.histplot(data=self.pop2_path_dict['u'][:,self.NT1], bins=100,stat='count',alpha=0.6,color= 'green')
        #   sns.kdeplot(self.pop2_path_dict['u'][:,self.NT1], color="green")
        #   plt.xlabel("$U_{T_1}^{(2)}$")
        #   plt.ylabel("$U_{T_1}^{(2)}$ Count")

        #   plt.subplot(133)
        #   plt.title("Distribution of $Y_{T_2}^{(2)}$")
        #   sns.histplot(data=self.pop2_path_dict['y'][idx0_y2,self.NT2], bins=100,stat='count',alpha=0.6,color= 'green')
        #   sns.histplot(data=self.pop2_path_dict['y'][idx1_y2,self.NT2], bins=100,stat='count',alpha=0.3,color= 'grey')
        #   sns.kdeplot(self.pop2_path_dict['y'][idx0_y2,self.NT2], color="green",label=('$Y_{T_2}^{(2)}\\rightarrow$ 0'))
        #   sns.kdeplot(self.pop2_path_dict['y'][idx1_y2,self.NT2], color="black",label=('$Y_{T_2}^{(2)}\\rightarrow$ 1'))
        #   plt.xlabel("$Y_{T_2}^{(2)}$")
        #   plt.ylabel("$Y_{T_2}^{(2)}$ Count")
        #   plt.legend()




## ===================================== Dumped ===================================== ##

# def get_forward_loss(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, main_models}
#   ## -------------------------------- P1 Params -------------------------------- ##
#   pi1=pop1_dict['GlobalParams'].pi
#   h1=pop1_dict['GlobalParams'].h
#   zeta1=pop1_dict['GlobalParams'].zeta
#   beta1=pop1_dict['GlobalParams'].beta
#   gamma1=pop1_dict['GlobalParams'].gamma
#   sigma1=pop1_dict['GlobalParams'].sigma
#   x1=pop1_dict['init_x']
#   c1=pop1_dict['init_c']
#   main_models1=pop1_dict['main_models']
#   dB1=pop1_dict['dB']
#   ## -------------------------------- P2 Params -------------------------------- ##
#   pi2=pop2_dict['GlobalParams'].pi
#   h2=pop2_dict['GlobalParams'].h
#   zeta2=pop2_dict['GlobalParams'].zeta
#   beta2=pop2_dict['GlobalParams'].beta
#   gamma2=pop2_dict['GlobalParams'].gamma
#   sigma2=pop2_dict['GlobalParams'].sigma
#   x2=pop2_dict['init_x']
#   c2=pop2_dict['init_c']
#   main_models2=pop2_dict['main_models']
#   dB2=pop2_dict['dB']
#   ## -------------------------------- Common Params -------------------------------- ##
#   K=pop1_dict['GlobalParams'].K
#   w=pop1_dict['GlobalParams'].w
#   dt=pop1_dict['GlobalParams'].dt
#   NT1=pop1_dict['GlobalParams'].NT1
#   NT2=pop1_dict['GlobalParams'].NT2
#   NumTrain=pop1_dict['GlobalParams'].NumTrain
#   device=pop1_dict['GlobalParams'].device
#   trick=pop1_dict['GlobalParams'].trick
#   loss_type=pop1_dict['GlobalParams'].loss_type
#   omiga1=(pi1/gamma1)/(pi1/gamma1+pi2/gamma2)
#   omiga2=(pi2/gamma2)/(pi1/gamma1+pi2/gamma2)

#   if trick=='no':
#     for j in range(0, NT2+1):
#       if j==0:
#         v1=main_models1.v0_model(x1).view(-1,1)
#         v2=main_models2.v0_model(x2).view(-1,1)
        
#         u1=main_models1.u0_model(x1).view(-1,1)
#         u2=main_models2.u0_model(x2).view(-1,1)
        
#         y1=main_models1.y0_model(x1).view(-1,1)
#         y2=main_models2.y0_model(x2).view(-1,1)
        
#         S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
#       else:
#         x1 =x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
#         x2 =x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

#         c1=c1+a1*dt
#         c2=c2+a2*dt

#         if j<=NT1:
#           zv1 = main_models1.zv_models[j-1](x1)
#           zv2 = main_models2.zv_models[j-1](x2)

#           zu1 = main_models1.zu_models[j-1](x1)
#           zu2 = main_models2.zu_models[j-1](x2)
          
#         zy1 = main_models1.zy_models[j-1](x1)
#         zy2 = main_models2.zy_models[j-1](x2)
        
#         v1=v1+zv1*dB1[:,j].view(-1,1) if j<=NT1 else v1
#         v2=v2+zv2*dB2[:,j].view(-1,1) if j<=NT1 else v2

#         u1=u1+zu1*dB1[:,j].view(-1,1) if j<=NT1 else u1
#         u2=u2+zu2*dB2[:,j].view(-1,1) if j<=NT1 else u2

#         y1=y1+zy1*dB1[:,j].view(-1,1)
#         y2=y2+zy2*dB2[:,j].view(-1,1)

#         S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
      
#       g1=(v1+u1)/zeta1 if j<=NT1 else y1/zeta1
#       g2=(v2+u2)/zeta2 if j<=NT1 else y2/zeta2

#       Gamma1=(v1+u1-S)/gamma1 if j<=NT1 else (y1-S)/gamma1
#       Gamma2=(v2+u2-S)/gamma2 if j<=NT1 else (y2-S)/gamma2

#       a1=((NT1-j)*(v1+u1)+(NT2-NT1)*y1)*dt/beta1 if j<=NT1 else (NT2-j)*y1*dt/beta1
#       a2=((NT1-j)*(v2+u2)+(NT2-NT1)*y2)*dt/beta2 if j<=NT1 else (NT2-j)*y2*dt/beta2

#       ## Freeze And Record T1 Values
#       if j==NT1:
#         x1_t1, x2_t1, y1_t1, y2_t1 = x1, x2, y1, y2 
  
#     loss_v1=Loss(pred=v1/w,targ=target_V(x_t1=x1_t1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type=loss_type)
#     loss_v2=Loss(pred=v2/w,targ=target_V(x_t1=x2_t1,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type=loss_type)
    
#     loss_u1=Loss(pred=u1/w,targ=target_U(x_t1=x1_t1,y_t1=y1_t1/w,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='MSELoss')
#     loss_u2=Loss(pred=u2/w,targ=target_U(x_t1=x2_t1,y_t1=y2_t1/w,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='MSELoss')
    
#     loss_y1=Loss(pred=y1/w,targ=target_Y(x_t1=x1_t1,x_t2=x1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type=loss_type)
#     loss_y2=Loss(pred=y2/w,targ=target_Y(x_t1=x2_t1,x_t2=x2,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type=loss_type)
    
#     loss=(loss_v1+loss_u1+loss_y1)+(loss_v2+loss_u2+loss_y2)
#     return loss.to(device)
    


# def get_target_path(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, main_models}
#   ## -------------------------------- P1 Params -------------------------------- ##
#   pi1=pop1_dict['GlobalParams'].pi
#   h1=pop1_dict['GlobalParams'].h
#   zeta1=pop1_dict['GlobalParams'].zeta
#   beta1=pop1_dict['GlobalParams'].beta
#   gamma1=pop1_dict['GlobalParams'].gamma
#   sigma1=pop1_dict['GlobalParams'].sigma
#   x1=pop1_dict['init_x']
#   c1=pop1_dict['init_c']
#   main_models1=pop1_dict['main_models']
#   dB1=pop1_dict['dB']
#   ## -------------------------------- P2 Params -------------------------------- ##
#   pi2=pop2_dict['GlobalParams'].pi
#   h2=pop2_dict['GlobalParams'].h
#   zeta2=pop2_dict['GlobalParams'].zeta
#   beta2=pop2_dict['GlobalParams'].beta
#   gamma2=pop2_dict['GlobalParams'].gamma
#   sigma2=pop2_dict['GlobalParams'].sigma
#   x2=pop2_dict['init_x']
#   c2=pop2_dict['init_c']
#   main_models2=pop2_dict['main_models']
#   dB2=pop2_dict['dB']
#   ## -------------------------------- Common Params -------------------------------- ##
#   w=pop1_dict['GlobalParams'].w
#   dt=pop1_dict['GlobalParams'].dt
#   NT1=pop1_dict['GlobalParams'].NT1
#   NT2=pop1_dict['GlobalParams'].NT2
#   NumTrain=pop1_dict['GlobalParams'].NumTrain
#   device=pop1_dict['GlobalParams'].device
#   trick=pop1_dict['GlobalParams'].trick
#   omiga1=(pi1/gamma1)/(pi1/gamma1+pi2/gamma2)
#   omiga2=(pi2/gamma2)/(pi1/gamma1+pi2/gamma2)
#   ## -------------------------------- Paths -------------------------------- ##
#   x1_path = torch.ones(NumTrain,NT2+1).to(device)
#   x2_path = torch.ones(NumTrain,NT2+1).to(device)

#   v1_path = torch.ones(NumTrain,NT2+1).to(device)
#   v2_path = torch.ones(NumTrain,NT2+1).to(device)

#   u1_path = torch.ones(NumTrain,NT2+1).to(device)
#   u2_path = torch.ones(NumTrain,NT2+1).to(device)

#   y1_path = torch.ones(NumTrain,NT2+1).to(device)
#   y2_path = torch.ones(NumTrain,NT2+1).to(device)

#   g1_path = torch.ones(NumTrain,NT2+1).to(device)
#   g2_path = torch.ones(NumTrain,NT2+1).to(device)
  
#   Gamma1_path = torch.ones(NumTrain,NT2+1).to(device)
#   Gamma2_path = torch.ones(NumTrain,NT2+1).to(device)
  
#   a1_path = torch.ones(NumTrain,NT2+1).to(device)
#   a2_path = torch.ones(NumTrain,NT2+1).to(device)

#   S_path=torch.ones(NT2+1).to(device)

#   with torch.inference_mode():
#     if trick=='no':
#       for j in range(0, NT2+1):
#         if j==0:
#           v1=main_models1.v0_model(x1).view(-1,1)
#           v2=main_models2.v0_model(x2).view(-1,1)
          
#           u1=main_models1.u0_model(x1).view(-1,1)
#           u2=main_models2.u0_model(x2).view(-1,1)
          
#           y1=main_models1.y0_model(x1).view(-1,1)
#           y2=main_models2.y0_model(x2).view(-1,1)

#         else:   ### j>0:
#           x1 =x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
#           x2 =x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

#           c1=c1+a1*dt
#           c2=c2+a2*dt

#           if j<=NT1:
#             zv1 = main_models1.zv_models[j-1](x1)
#             zv2 = main_models2.zv_models[j-1](x2)

#             zu1 = main_models1.zu_models[j-1](x1)
#             zu2 = main_models2.zu_models[j-1](x2)
            
#           zy1 = main_models1.zy_models[j-1](x1)
#           zy2 = main_models2.zy_models[j-1](x2)
          
#           v1=v1+zv1*dB1[:,j].view(-1,1) if j<=NT1 else v1
#           v2=v2+zv2*dB2[:,j].view(-1,1) if j<=NT1 else v2

#           u1=u1+zu1*dB1[:,j].view(-1,1) if j<=NT1 else u1
#           u2=u2+zu2*dB2[:,j].view(-1,1) if j<=NT1 else u2

#           y1=y1+zy1*dB1[:,j].view(-1,1)
#           y2=y2+zy2*dB2[:,j].view(-1,1)

#         S=(omiga1*((v1+u1).mean())+omiga2*((v2+u2).mean())) if j<=NT1 else (omiga1*(y1.mean())+omiga2*(y2.mean()))
        
#         g1=(v1+u1)/zeta1 if j<=NT1 else y1/zeta1
#         g2=(v2+u2)/zeta2 if j<=NT1 else y2/zeta2

#         Gamma1=(v1+u1-S)/gamma1 if j<=NT1 else (y1-S)/gamma1
#         Gamma2=(v2+u2-S)/gamma2 if j<=NT1 else (y2-S)/gamma2

#         a1=((NT1-j)*(v1+u1)+(NT2-NT1)*y1)*dt/beta1 if j<=NT1 else (NT2-j)*dt*y1/beta1
#         a2=((NT1-j)*(v2+u2)+(NT2-NT1)*y2)*dt/beta2 if j<=NT1 else (NT2-j)*dt*y2/beta2
#         ## -------------------------------- Record Paths -------------------------------- ##
#         x1_path[:,j] = x1.squeeze()
#         x2_path[:,j] = x2.squeeze()
        
#         v1_path[:,j] = v1.squeeze()
#         v2_path[:,j] = v2.squeeze()

#         u1_path[:,j] = u1.squeeze()
#         u2_path[:,j] = u2.squeeze()

#         y1_path[:,j] = y1.squeeze()
#         y2_path[:,j] = y2.squeeze()

#         g1_path[:,j] = g1.squeeze()
#         g2_path[:,j] = g2.squeeze()

#         Gamma1_path[:,j] = Gamma1.squeeze()
#         Gamma2_path[:,j] = Gamma2.squeeze()

#         a1_path[:,j] = a1.squeeze()
#         a2_path[:,j] = a2.squeeze()

#         S_path[j]=S
