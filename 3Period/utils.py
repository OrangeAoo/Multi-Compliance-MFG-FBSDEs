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
  Nsteps=GlobalParams.NT3+1
  device=GlobalParams.device
  dB = np.sqrt(dt) * np.random.randn(Npaths, Nsteps)
  dB = torch.FloatTensor(dB).to(device)
  return dB

def target(x_end,GlobalParams,target_type=None, device = None):
  delta=GlobalParams.delta
  K=GlobalParams.K
  device=GlobalParams.device if device is None else torch.device(device)
  if target_type==None:
    target_type=GlobalParams.target_type

  if target_type=='sigmoid':# and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (torch.sigmoid((K-x_end)/delta)).to(device)
  
  if target_type=='indicator':# or (GlobalParams.target_type=='indicator' and (GlobalParams.trick=='no' or GlobalParams.trick=='clamp')):
    return (torch.where(x_end-K<0,1.0,0.0)).to(device)
  
  else:
      print("Please check whether 'target_type' matches 'trick' :)")

# Calculate Loss
def Loss(pred,targ,GlobalParams, loss_type=None):
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

def move_to_cpu(vars: dict) -> dict:
    return {k:v.cpu() for k,v in vars.items()}
    

# Forward Loss
def get_forward_loss(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, agents}
  ## -------------------------------- P1 Params -------------------------------- ##
  pi1=pop1_dict['GlobalParams'].pi
  h1=pop1_dict['GlobalParams'].h
  zeta1=pop1_dict['GlobalParams'].zeta
  beta1=pop1_dict['GlobalParams'].beta
  gamma1=pop1_dict['GlobalParams'].gamma
  # q1=pop1_dict['GlobalParams'].q     
  sigma1=pop1_dict['GlobalParams'].sigma
  agents1=pop1_dict['agents']
  dB1=pop1_dict['dB']
  x_agents1=pop1_dict['init_x']
  c_agents1=pop1_dict['init_c']
  ## -------------------------------- P2 Params -------------------------------- ##
  pi2=pop2_dict['GlobalParams'].pi
  h2=pop2_dict['GlobalParams'].h
  zeta2=pop2_dict['GlobalParams'].zeta
  beta2=pop2_dict['GlobalParams'].beta
  gamma2=pop2_dict['GlobalParams'].gamma
  # q2=pop2_dict['GlobalParams'].q    
  sigma2=pop2_dict['GlobalParams'].sigma
  agents2=pop2_dict['agents']
  dB2=pop2_dict['dB']
  x_agents2=pop2_dict['init_x']
  c_agents2=pop2_dict['init_c']
  ## -------------------------------- Common Params -------------------------------- ##
  w=pop1_dict['GlobalParams'].w
  K=pop1_dict['GlobalParams'].K
  dt=pop1_dict['GlobalParams'].dt
  NT1=pop1_dict['GlobalParams'].NT1
  NT2=pop1_dict['GlobalParams'].NT2
  NT3=pop1_dict['GlobalParams'].NT3
  device=pop1_dict['GlobalParams'].device
  trick=pop1_dict['GlobalParams'].trick
  loss_type=pop1_dict['GlobalParams'].loss_type
  omiga1=(pi1/gamma1)/(pi1/gamma1+pi2/gamma2)
  omiga2=(pi2/gamma2)/(pi1/gamma1+pi2/gamma2)
  
  # if trick=='logit': ## use yx_tilde=logit(yx), yx=sigmoid(yx_tilde), dyx=zx*(1-yx)*yx*dB
  #   for j in range(0, NT3+1):
  #     if j==0:  # @ 0 --> init for Prd1&2
  #       v1_tilde=w*agents1.v0_model(x1).view(-1,1)
  #       v2_tilde=w*agents2.v0_model(x2).view(-1,1)
  #       v1=w*torch.sigmoid(v1_tilde/w).view(-1,1).to(device)
  #       v2=w*torch.sigmoid(v2_tilde/w).view(-1,1).to(device)

  #       u1_tilde=w*agents1.u0_model(x1).view(-1,1)
  #       u2_tilde=w*agents2.u0_model(x2).view(-1,1)
  #       u1=w*torch.sigmoid(u1_tilde/w).view(-1,1).to(device)
  #       u2=w*torch.sigmoid(u2_tilde/w).view(-1,1).to(device)

  #       y1_tilde=w*agents1.y0_model(x1).view(-1,1)
  #       y2_tilde=w*agents2.y0_model(x2).view(-1,1)
  #       y1=w*torch.sigmoid(y1_tilde/w).view(-1,1).to(device)
  #       y2=w*torch.sigmoid(y2_tilde/w).view(-1,1).to(device)
        
  #     else:
  #       x1 = x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
  #       x2 = x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

  #       c1=c1+a1*dt
  #       c2=c2+a2*dt

  #       if j<=NT1:
  #         zv1 = agents1.zv_models[j-1](x1)
  #         zv2 = agents2.zv_models[j-1](x2)

  #         zu1 = agents1.zu_models[j-1](x1)
  #         zu2 = agents2.zu_models[j-1](x2)
          
  #       zy1 = agents1.zy_models[j-1](x1)
  #       zy2 = agents2.zy_models[j-1](x2)
        
  #       v1_tilde=v1_tilde+(zv1**2)*(v1-w/2)*dt+w*zv1*dB1[:,j].view(-1,1) if j<=NT1 else v1_tilde
  #       v2_tilde=v2_tilde+(zv2**2)*(v2-w/2)*dt+w*zv2*dB2[:,j].view(-1,1) if j<=NT1 else v2_tilde
  #       v1=w*torch.sigmoid(v1_tilde/w).view(-1,1).to(device) if j<=NT1 else v1
  #       v2=w*torch.sigmoid(v2_tilde/w).view(-1,1).to(device) if j<=NT1 else v2

  #       u1_tilde=u1_tilde+(zu1**2)*(u1-w/2)*dt+w*zu1*dB1[:,j].view(-1,1) if j<=NT1 else u1_tilde
  #       u2_tilde=u2_tilde+(zu2**2)*(u2-w/2)*dt+w*zu2*dB2[:,j].view(-1,1) if j<=NT1 else u2_tilde
  #       u1=w*torch.sigmoid(u1_tilde/w).view(-1,1).to(device) if j<=NT1 else u1
  #       u2=w*torch.sigmoid(u2_tilde/w).view(-1,1).to(device) if j<=NT1 else u2

  #       y1_tilde=y1_tilde+(zy1**2)*(y1-w/2)*dt+w*zy1*dB1[:,j].view(-1,1)
  #       y2_tilde=y2_tilde+(zy2**2)*(y2-w/2)*dt+w*zy2*dB2[:,j].view(-1,1)
  #       y1=(w*torch.sigmoid(y1_tilde/w).view(-1,1)).to(device)
  #       y2=(w*torch.sigmoid(y2_tilde/w).view(-1,1)).to(device)
      
  #     if j<=NT1:  
  #       S=(omiga1*((q1*(NT2-NT1)*dt*v1+v1+u1).mean()-q1*(NT2-j)*dt)+\
  #        omiga2*((q2*(NT2-NT1)*dt*v2+v2+u2).mean()-q2*(NT2-j)*dt))  
  #     else: 
  #        S=(omiga1*(y1.mean()-q1*(NT2-j)*dt)+\
  #           omiga2*(y2.mean()-q2*(NT2-j)*dt))

  #     g1=(v1+u1+q1*(NT2-NT1)*dt*v1-q1*(NT2-j)*dt)/zeta1 if j<=NT1 else (y1-q1*(NT2-j)*dt)/zeta1
  #     g2=(v2+u2+q2*(NT2-NT1)*dt*v2-q2*(NT2-j)*dt)/zeta2 if j<=NT1 else (y2-q2*(NT2-j)*dt)/zeta2

  #     Gamma1=(v1+u1+q1*(NT2-NT1)*dt*v1-S-q1*(NT2-j)*dt)/gamma1 if j<=NT1 else (y1-S-q1*(NT2-j)*dt)/gamma1
  #     Gamma2=(v2+u2+q2*(NT2-NT1)*dt*v2-S-q2*(NT2-j)*dt)/gamma2 if j<=NT1 else (y2-S-q2*(NT2-j)*dt)/gamma2

  #     a1=((NT1-j)*dt*(v1+u1)+(NT2-NT1)*dt*(y1+q1*(NT1-j)*dt*v1)-q1*((NT2-j)*dt)**2/2)/beta1 if j<=NT1 else ((NT2-j)*y1*dt-q1*((NT2-j)*dt)**2/2)/beta1  #*dt= 0.02
  #     a2=((NT1-j)*dt*(v2+u2)+(NT2-NT1)*dt*(y2+q2*(NT1-j)*dt*v2)-q2*((NT2-j)*dt)**2/2)/beta2 if j<=NT1 else ((NT2-j)*y2*dt-q2*((NT2-j)*dt)**2/2)/beta2

  #     if j==NT1: # @ NT1 --> Loss for Prd1: loss_v, loss_u
  #       x1_t1, x2_t1, y1_t1, y2_t1 = x1, x2, y1, y2 
    
  #       loss_v1=Loss(pred=v1_tilde/w,targ=target_V(x_t1=x1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
  #       loss_v2=Loss(pred=v2_tilde/w,targ=target_V(x_t1=x2,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
        
  #       loss_u1=Loss(pred=u1_tilde/w,targ=target_U(x_t1=x1,y_t1=y1/w,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='MSELoss')
  #       loss_u2=Loss(pred=u2_tilde/w,targ=target_U(x_t1=x2,y_t1=y2/w,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='MSELoss')
        
  #       x1 = nn.ReLU()(x1-K)  # @ NT1: hand in min(K,xt1)
  #       x2 = nn.ReLU()(x2-K)
  #   # @ NT2 --> Loss for Prd2: loss_y
  #   loss_y1=Loss(pred=y1_tilde/w,targ=target_Y(x_t2=x1,GlobalParams=pop1_dict['GlobalParams']),GlobalParams=pop1_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
  #   loss_y2=Loss(pred=y2_tilde/w,targ=target_Y(x_t2=x2,GlobalParams=pop2_dict['GlobalParams']),GlobalParams=pop2_dict['GlobalParams'],loss_type='BCEWithLogitsLoss')
    
  #   loss=(loss_v1+loss_u1+loss_y1)+(loss_v2+loss_u2+loss_y2)
  #   return loss
  
  if trick=='clamp':  ## use dy=-zy*(1-y)*y*dB ONLY
    for j in range(0, NT3+1):
      if j==0:  # @ 0 --> init for Prd1&2
        y1_agents1=w*agents1.y1_0_model(x_agents1).view(-1,1)#.clamp(min=0,max=w)
        y12_agents1=w*agents1.y12_0_model(x_agents1).view(-1,1)#.clamp(min=0,max=w)
        y13_agents1=w*agents1.y13_0_model(x_agents1).view(-1,1)#.clamp(min=0,max=w)
        y2_agents1=w*agents1.y2_0_model(x_agents1).view(-1,1)
        y23_agents1=w*agents1.y23_0_model(x_agents1).view(-1,1)
        y3_agents1=w*agents1.y3_0_model(x_agents1).view(-1,1)
        
        y1_agents2=w*agents2.y1_0_model(x_agents2).view(-1,1)#.clamp(min=0,max=w)
        y12_agents2=w*agents2.y12_0_model(x_agents2).view(-1,1)#.clamp(min=0,max=w)
        y13_agents2=w*agents2.y13_0_model(x_agents2).view(-1,1)#.clamp(min=0,max=w)
        y2_agents2=w*agents2.y2_0_model(x_agents2).view(-1,1)
        y23_agents2=w*agents2.y23_0_model(x_agents2).view(-1,1)
        y3_agents2=w*agents2.y3_0_model(x_agents2).view(-1,1)
        

      elif j>0:
        x_agents1 =x_agents1+ (h1+g_agents1+Gamma_agents1+c_agents1)*dt+sigma1*dB1[:,j].view(-1,1)
        x_agents2 =x_agents2+ (h2+g_agents2+Gamma_agents2+c_agents2)*dt+sigma2*dB2[:,j].view(-1,1)        

        c_agents1=c_agents1+a_agents1*dt
        c_agents2=c_agents2+a_agents2*dt

        if j<=NT2:
          if j<=NT1: 
            zy1_agents1 = agents1.zy1_models[j-1](x_agents1)
            zy12_agents1 = agents1.zy12_models[j-1](x_agents1)
            zy13_agents1 = agents1.zy13_models[j-1](x_agents1)

            zy1_agents2 = agents2.zy1_models[j-1](x_agents1)
            zy12_agents2 = agents2.zy12_models[j-1](x_agents1)
            zy13_agents2 = agents2.zy13_models[j-1](x_agents1)
          
          zy2_agents1 = agents1.zy2_models[j-1](x_agents1)
          zy23_agents1 = agents1.zy23_models[j-1](x_agents1)
        
          zy2_agents2 = agents2.zy2_models[j-1](x_agents1)
          zy23_agents2 = agents2.zy23_models[j-1](x_agents1)

        zy3_agents1 = agents1.zy3_models[j-1](x_agents1)
        zy3_agents2 = agents2.zy3_models[j-1](x_agents1)
        
        # ---------------------- update y ---------------------- #
        y1_agents1=((y1_agents1+zy1_agents1*dB1[:,j].view(-1,1)) if j<=NT1 else y1_agents1).clamp(min=0,max=w)  ## y1_agents1*(1-y1_agents1/w)*
        y12_agents1=((y12_agents1+zy12_agents1*dB1[:,j].view(-1,1)) if j<=NT1 else y12_agents1).clamp(min=0,max=w)
        y13_agents1=((y13_agents1+zy13_agents1*dB1[:,j].view(-1,1)) if j<=NT1 else y13_agents1).clamp(min=0,max=w)
        y2_agents1=((y2_agents1+zy2_agents1*dB1[:,j].view(-1,1)) if j<=NT2 else y2_agents1).clamp(min=0,max=w)  ## y2_agents1*(1-y2_agents1/w)*
        y23_agents1=((y23_agents1+zy23_agents1*dB1[:,j].view(-1,1)) if j<=NT2 else y23_agents1).clamp(min=0,max=w)  ## y23_agents1*(1-y23_agents1/w)*
        y3_agents1=((y3_agents1+zy3_agents1*dB1[:,j].view(-1,1))).clamp(min=0,max=w)  ## y3_agents1*(1-y3_agents1/w)*
      
        y1_agents2=((y1_agents2+zy1_agents2*dB1[:,j].view(-1,1)) if j<=NT1 else y1_agents2).clamp(min=0,max=w)  ## y1_agents2*(1-y1_agents2/w)*
        y12_agents2=((y12_agents2+zy12_agents2*dB1[:,j].view(-1,1)) if j<=NT1 else y12_agents2).clamp(min=0,max=w)
        y13_agents2=((y13_agents2+zy13_agents2*dB1[:,j].view(-1,1)) if j<=NT1 else y13_agents2).clamp(min=0,max=w)
        y2_agents2=((y2_agents2+zy2_agents2*dB1[:,j].view(-1,1)) if j<=NT2 else y2_agents2).clamp(min=0,max=w)  ## y2_agents2*(1-y2_agents2/w)*
        y23_agents2=((y23_agents2+zy23_agents2*dB1[:,j].view(-1,1)) if j<=NT2 else y23_agents2).clamp(min=0,max=w)  ## y23_agents2*(1-y23_agents2/w)*
        y3_agents2=((y3_agents2+zy3_agents2*dB1[:,j].view(-1,1))).clamp(min=0,max=w)  ## y3_agents2*(1-y3_agents2/w)*


      S=(omiga1*((y1_agents1+y12_agents1+y13_agents1).mean())+\
         omiga2*((y1_agents2+y12_agents2+y13_agents2).mean())) * (j<NT1)+\
        (omiga1*((y2_agents1+y23_agents1).mean())+\
         omiga2*((y2_agents2+y23_agents2).mean()))*(j>=NT1 and j<NT2)+\
        (omiga1*((y3_agents1).mean())+\
         omiga2*((y3_agents2).mean())*(j>=NT2))

      g_agents1=((y1_agents1+y12_agents1+y13_agents1)*(j<NT1)+\
                 (y2_agents1+y23_agents1)*(j>=NT1 and j<NT2)+\
                 (y3_agents1)*(j>=NT2)
                )/zeta1
      g_agents2=((y1_agents2+y12_agents2+y13_agents2)*(j<NT1)+\
                 (y2_agents2+y23_agents2)*(j>=NT1 and j<NT2)+\
                 (y3_agents2)*(j>=NT2)
                )/zeta2

      Gamma_agents1 =((y1_agents1+y12_agents1+y13_agents1-S)*(j<NT1)+\
                      (y2_agents1+y23_agents1-S)*(j>=NT1 and j<NT2)+\
                      (y3_agents1-S)*(j>=NT2)
                      )/gamma1
      Gamma_agents2 =((y1_agents2+y12_agents2+y13_agents2-S)*(j<NT1)+\
                      (y2_agents2+y23_agents2-S)*(j>=NT1 and j<NT2)+\
                      (y3_agents2-S)*(j>=NT2)
                      )/gamma2
      
      a_agents1 =(((NT3-NT2)*y3_agents1 + (NT2-NT1)*(y2_agents1+y23_agents1) + (NT1-j)*(y1_agents1+y12_agents1+y13_agents1)) * (j<NT1)+\
                  ((NT3-NT2)*y3_agents1 + (NT2-j)*(y2_agents1+y23_agents1))*(j>=NT1 and j<NT2)+\
                  ((NT3-j)*y3_agents1)*(j>=NT2)
                  ) * dt/beta1
      a_agents2 =(((NT3-NT2)*y3_agents2 + (NT2-NT1)*(y2_agents2+y23_agents2) + (NT1-j)*(y1_agents2+y12_agents2+y13_agents2)) * (j<NT1)+\
                  ((NT3-NT2)*y3_agents2 + (NT2-j)*(y2_agents2+y23_agents2))*(j>=NT1 and j<NT2)+\
                  ((NT3-j)*y3_agents2)*(j>=NT2)
                  ) * dt/beta2

      if j==NT1:  # @ NT1 / NT2 --> Loss for Prd1/2: loss_v, loss_u; Clearance: x <- relu(x-k), hand in min(x,K)
         default_T1_agents1 = target(x_end=x_agents1,GlobalParams=pop1_dict['GlobalParams'])
         default_T1_agents2 = target(x_end=x_agents2,GlobalParams=pop2_dict['GlobalParams'])
         x_agents1 = nn.ReLU()(x_agents1-K)  # @ NT1: hand in min(K,xt1)
         x_agents2 = nn.ReLU()(x_agents2-K)
      elif j==NT2:  # @ NT1 / NT2 --> Loss for Prd1/2: loss_v, loss_u; Clearance: x <- relu(x-k), hand in min(x,K)
        default_T2_agents1 = target(x_end=x_agents1,GlobalParams=pop1_dict['GlobalParams'])
        default_T2_agents2 = target(x_end=x_agents2,GlobalParams=pop2_dict['GlobalParams'])
        x_agents1 = nn.ReLU()(x_agents1-K)  # @ NT1: hand in min(K,xt1)
        x_agents2 = nn.ReLU()(x_agents2-K)   
      elif j==NT3:
        default_T3_agents1 = target(x_end=x_agents1,GlobalParams=pop1_dict['GlobalParams'])
        default_T3_agents2 = target(x_end=x_agents2,GlobalParams=pop2_dict['GlobalParams'])
      
    # ================= Loss ================= #
    # @ NT1 --> Loss for Prd1:
    loss_y1_agents1=Loss(pred=y1_agents1/w,
                          targ=default_T1_agents1,
                          GlobalParams=pop1_dict['GlobalParams'],
                          loss_type=loss_type)
    loss_y12_agents1=Loss(pred=y12_agents1/w,
                          targ=(1-default_T1_agents1)*default_T2_agents1,
                          GlobalParams=pop1_dict['GlobalParams'],
                          loss_type=loss_type)
    loss_y13_agents1=Loss(pred=y13_agents1/w,
                          targ=(1-default_T1_agents1)*default_T3_agents1,
                          GlobalParams=pop1_dict['GlobalParams'],
                          loss_type=loss_type)
    
    loss_y1_agents2=Loss(pred=y1_agents2/w,
                          targ=default_T1_agents2,
                          GlobalParams=pop2_dict['GlobalParams'],
                          loss_type=loss_type)
    loss_y12_agents2=Loss(pred=y12_agents2/w,
                          targ=(1-default_T1_agents2)*default_T2_agents2,
                          GlobalParams=pop2_dict['GlobalParams'],
                          loss_type=loss_type)
    loss_y13_agents2=Loss(pred=y13_agents2/w,
                          targ=(1-default_T1_agents2)*default_T3_agents2,
                          GlobalParams=pop2_dict['GlobalParams'],
                          loss_type=loss_type)
    # @ NT2 --> Loss for Prd2:
    loss_y2_agents1 =Loss(pred=y2_agents1/w,
                          targ=default_T2_agents1,
                          GlobalParams=pop1_dict['GlobalParams'],
                          loss_type=loss_type)
    loss_y23_agents1 =Loss(pred=y23_agents1/w,
                          targ=(1-default_T2_agents1)*default_T3_agents1,
                          GlobalParams=pop1_dict['GlobalParams'],
                          loss_type=loss_type)
    
    loss_y2_agents2 =Loss(pred=y2_agents2/w,
                          targ=default_T2_agents2,
                          GlobalParams=pop2_dict['GlobalParams'],
                          loss_type=loss_type)
    loss_y23_agents2 =Loss(pred=y23_agents2/w,
                          targ=(1-default_T2_agents2)*default_T3_agents2,
                          GlobalParams=pop2_dict['GlobalParams'],
                          loss_type=loss_type)
    # @ NT3 --> Loss for Prd3:
    loss_y3_agents1 =Loss(pred=y3_agents1/w,
                          targ=default_T3_agents1,
                          GlobalParams=pop1_dict['GlobalParams'],
                          loss_type=loss_type)
    loss_y3_agents2 =Loss(pred=y3_agents2/w,
                          targ=default_T3_agents2,
                          GlobalParams=pop2_dict['GlobalParams'],
                          loss_type=loss_type)
    
    loss=(loss_y1_agents1+loss_y12_agents1+loss_y13_agents1+\
          loss_y2_agents1+loss_y23_agents1+\
          loss_y3_agents1)+\
         (loss_y1_agents2+loss_y12_agents2+loss_y13_agents2+\
          loss_y2_agents2+loss_y23_agents2+\
          loss_y3_agents2)
    return loss
    
def get_target_path(pop1_dict, pop2_dict):# pop_dict={dB, init_x,init_c, GlobalParams, agents}
  ## -------------------------------- P1 Params -------------------------------- ##
  pi1=pop1_dict['GlobalParams'].pi
  h1=pop1_dict['GlobalParams'].h
  zeta1=pop1_dict['GlobalParams'].zeta
  beta1=pop1_dict['GlobalParams'].beta
  gamma1=pop1_dict['GlobalParams'].gamma
  # q1=pop1_dict['GlobalParams'].q     
  sigma1=pop1_dict['GlobalParams'].sigma
  agents1=pop1_dict['agents']
  dB1=pop1_dict['dB']
  x_agents1=pop1_dict['init_x']
  c_agents1=pop1_dict['init_c']
  ## -------------------------------- P2 Params -------------------------------- ##
  pi2=pop2_dict['GlobalParams'].pi
  h2=pop2_dict['GlobalParams'].h
  zeta2=pop2_dict['GlobalParams'].zeta
  beta2=pop2_dict['GlobalParams'].beta
  gamma2=pop2_dict['GlobalParams'].gamma
  # q2=pop2_dict['GlobalParams'].q    
  sigma2=pop2_dict['GlobalParams'].sigma
  agents2=pop2_dict['agents']
  dB2=pop2_dict['dB']
  x_agents2=pop2_dict['init_x']
  c_agents2=pop2_dict['init_c']
  ## -------------------------------- Common Params -------------------------------- ##
  w=pop1_dict['GlobalParams'].w
  K=pop1_dict['GlobalParams'].K
  dt=pop1_dict['GlobalParams'].dt
  NT1=pop1_dict['GlobalParams'].NT1
  NT2=pop1_dict['GlobalParams'].NT2
  NT3=pop1_dict['GlobalParams'].NT3
  NumTrain=pop1_dict['GlobalParams'].NumTrain
  device=pop1_dict['GlobalParams'].device
  trick=pop1_dict['GlobalParams'].trick
  omiga1=(pi1/gamma1)/(pi1/gamma1+pi2/gamma2)
  omiga2=(pi2/gamma2)/(pi1/gamma1+pi2/gamma2)
  ## -------------------------------- Paths -------------------------------- ##
  x_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  x_agents2_path = torch.ones(NumTrain,NT3+1).to(device)

  y1_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  y12_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  y13_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  y2_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  y23_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  y3_agents1_path = torch.ones(NumTrain,NT3+1).to(device)

  y1_agents2_path = torch.ones(NumTrain,NT3+1).to(device)
  y12_agents2_path = torch.ones(NumTrain,NT3+1).to(device)
  y13_agents2_path = torch.ones(NumTrain,NT3+1).to(device)
  y2_agents2_path = torch.ones(NumTrain,NT3+1).to(device)
  y23_agents2_path = torch.ones(NumTrain,NT3+1).to(device)
  y3_agents2_path = torch.ones(NumTrain,NT3+1).to(device)

  g_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  g_agents2_path = torch.ones(NumTrain,NT3+1).to(device)
  
  Gamma_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  Gamma_agents2_path = torch.ones(NumTrain,NT3+1).to(device)
  
  a_agents1_path = torch.ones(NumTrain,NT3+1).to(device)
  a_agents2_path = torch.ones(NumTrain,NT3+1).to(device)

  S_path=torch.ones(NT3+1).to(device)

  with torch.inference_mode():
    if trick=='logit': ## use yx_tilde=logit(-yx), yx=-sigmoid(yx_tilde), dyx=-zx*(1+yx)*yx*dB
      for j in range(0, NT2+1):
        if j==0:
          v1_tilde=w*agents1.v0_model(x1).view(-1,1)
          v2_tilde=w*agents2.v0_model(x2).view(-1,1)
          v1=w*torch.sigmoid(v1_tilde/w).view(-1,1).to(device)
          v2=w*torch.sigmoid(v2_tilde/w).view(-1,1).to(device)

          u1_tilde=w*agents1.u0_model(x1).view(-1,1)
          u2_tilde=w*agents2.u0_model(x2).view(-1,1)
          u1=w*torch.sigmoid(u1_tilde/w).view(-1,1).to(device)
          u2=w*torch.sigmoid(u2_tilde/w).view(-1,1).to(device)

          y1_tilde=w*agents1.y0_model(x1).view(-1,1)
          y2_tilde=w*agents2.y0_model(x2).view(-1,1)
          y1=w*torch.sigmoid(y1_tilde/w).view(-1,1).to(device)
          y2=w*torch.sigmoid(y2_tilde/w).view(-1,1).to(device)
          
        else:
          x1 =x1+ (h1+g1+Gamma1+c1)*dt+sigma1*dB1[:,j].view(-1,1)
          x2 =x2+ (h2+g2+Gamma2+c2)*dt+sigma2*dB2[:,j].view(-1,1)        

          c1=c1+a1*dt
          c2=c2+a2*dt

          if j<=NT1:
            zv1 = agents1.zv_models[j-1](x1)
            zv2 = agents2.zv_models[j-1](x2)

            zu1 = agents1.zu_models[j-1](x1)
            zu2 = agents2.zu_models[j-1](x2)
            
          zy1 = agents1.zy_models[j-1](x1)
          zy2 = agents2.zy_models[j-1](x2)
          
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

        S=(omiga1*((q1*(NT2-NT1)*dt*v1+v1+u1).mean()-q1*(NT2-j)*dt)+omiga2*((q2*(NT2-NT1)*dt*v2+v2+u2).mean()-q2*(NT2-j)*dt)) if j<=NT1 else (omiga1*(y1.mean()-q1*(NT2-j)*dt)+omiga2*(y2.mean()-q2*(NT2-j)*dt))

        g1=(v1+u1+q1*(NT2-NT1)*dt*v1-q1*(NT2-j)*dt)/zeta1 if j<=NT1 else (y1-q1*(NT2-j)*dt)/zeta1
        g2=(v2+u2+q2*(NT2-NT1)*dt*v2-q2*(NT2-j)*dt)/zeta2 if j<=NT1 else (y2-q2*(NT2-j)*dt)/zeta2

        Gamma1=(v1+u1+q1*(NT2-NT1)*dt*v1-S-q1*(NT2-j)*dt)/gamma1 if j<=NT1 else (y1-S-q1*(NT2-j)*dt)/gamma1
        Gamma2=(v2+u2+q2*(NT2-NT1)*dt*v2-S-q2*(NT2-j)*dt)/gamma2 if j<=NT1 else (y2-S-q2*(NT2-j)*dt)/gamma2

        a1=((NT1-j)*dt*(v1+u1)+(NT2-NT1)*dt*(y1+q1*(NT1-j)*dt*v1)-q1*((NT2-j)*dt)**2/2)/beta1 if j<=NT1 else ((NT2-j)*y1*dt-q1*((NT2-j)*dt)**2/2)/beta1  #*dt= 0.02
        a2=((NT1-j)*dt*(v2+u2)+(NT2-NT1)*dt*(y2+q2*(NT1-j)*dt*v2)-q2*((NT2-j)*dt)**2/2)/beta2 if j<=NT1 else ((NT2-j)*y2*dt-q2*((NT2-j)*dt)**2/2)/beta2

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

        S_path[j]=S.squeeze()
        ## -------------------------------- ------------ -------------------------------- ##
        if j==NT1:  # @ NT1 --> hand in min(K,xt1); init for Prd2 (with remaining inventory)
          x1 = nn.ReLU()(x1-K)   
          x2 = nn.ReLU()(x2-K)

      if trick=='clamp':  ## use dy=-zy*(1-y)*y*dB ONLY
        for j in range(0, NT3+1):
          if j==0:  # @ 0 --> init for Prd1&2
            y1_agents1=w*agents1.y1_0_model(x_agents1).view(-1,1)#.clamp(min=0,max=w)
            y12_agents1=w*agents1.y12_0_model(x_agents1).view(-1,1)#.clamp(min=0,max=w)
            y13_agents1=w*agents1.y13_0_model(x_agents1).view(-1,1)#.clamp(min=0,max=w)
            y2_agents1=w*agents1.y2_0_model(x_agents1).view(-1,1)
            y23_agents1=w*agents1.y23_0_model(x_agents1).view(-1,1)
            y3_agents1=w*agents1.y3_0_model(x_agents1).view(-1,1)
            
            y1_agents2=w*agents2.y1_0_model(x_agents2).view(-1,1)#.clamp(min=0,max=w)
            y12_agents2=w*agents2.y12_0_model(x_agents2).view(-1,1)#.clamp(min=0,max=w)
            y13_agents2=w*agents2.y13_0_model(x_agents2).view(-1,1)#.clamp(min=0,max=w)
            y2_agents2=w*agents2.y2_0_model(x_agents2).view(-1,1)
            y23_agents2=w*agents2.y23_0_model(x_agents2).view(-1,1)
            y3_agents2=w*agents2.y3_0_model(x_agents2).view(-1,1)
            

          elif j>0:
            x_agents1 =x_agents1+ (h1+g_agents1+Gamma_agents1+c_agents1)*dt+sigma1*dB1[:,j].view(-1,1)
            x_agents2 =x_agents2+ (h2+g_agents2+Gamma_agents2+c_agents2)*dt+sigma2*dB2[:,j].view(-1,1)        

            c_agents1=c_agents1+a_agents1*dt
            c_agents2=c_agents2+a_agents2*dt

            if j<=NT2:
              if j<=NT1: 
                zy1_agents1 = agents1.zy1_models[j-1](x_agents1)
                zy12_agents1 = agents1.zy12_models[j-1](x_agents1)
                zy13_agents1 = agents1.zy13_models[j-1](x_agents1)

                zy1_agents2 = agents2.zy1_models[j-1](x_agents1)
                zy12_agents2 = agents2.zy12_models[j-1](x_agents1)
                zy13_agents2 = agents2.zy13_models[j-1](x_agents1)
              
              zy2_agents1 = agents1.zy2_models[j-1](x_agents1)
              zy23_agents1 = agents1.zy23_models[j-1](x_agents1)
            
              zy2_agents2 = agents2.zy2_models[j-1](x_agents1)
              zy23_agents2 = agents2.zy23_models[j-1](x_agents1)

            zy3_agents1 = agents1.zy3_models[j-1](x_agents1)
            zy3_agents2 = agents2.zy3_models[j-1](x_agents1)
            
            # ---------------------- update y ---------------------- #
            y1_agents1=((y1_agents1+zy1_agents1*dB1[:,j].view(-1,1)) if j<=NT1 else y1_agents1).clamp(min=0,max=w)  ## y1_agents1*(1-y1_agents1/w)*
            y12_agents1=((y12_agents1+zy12_agents1*dB1[:,j].view(-1,1)) if j<=NT1 else y12_agents1).clamp(min=0,max=w)
            y13_agents1=((y13_agents1+zy13_agents1*dB1[:,j].view(-1,1)) if j<=NT1 else y13_agents1).clamp(min=0,max=w)
            y2_agents1=((y2_agents1+zy2_agents1*dB1[:,j].view(-1,1)) if j<=NT2 else y2_agents1).clamp(min=0,max=w)  ## y2_agents1*(1-y2_agents1/w)*
            y23_agents1=((y23_agents1+zy23_agents1*dB1[:,j].view(-1,1)) if j<=NT2 else y23_agents1).clamp(min=0,max=w)  ## y23_agents1*(1-y23_agents1/w)*
            y3_agents1=((y3_agents1+zy3_agents1*dB1[:,j].view(-1,1))).clamp(min=0,max=w)  ## y3_agents1*(1-y3_agents1/w)*
          
            y1_agents2=((y1_agents2+zy1_agents2*dB1[:,j].view(-1,1)) if j<=NT1 else y1_agents2).clamp(min=0,max=w)  ## y1_agents2*(1-y1_agents2/w)*
            y12_agents2=((y12_agents2+zy12_agents2*dB1[:,j].view(-1,1)) if j<=NT1 else y12_agents2).clamp(min=0,max=w)
            y13_agents2=((y13_agents2+zy13_agents2*dB1[:,j].view(-1,1)) if j<=NT1 else y13_agents2).clamp(min=0,max=w)
            y2_agents2=((y2_agents2+zy2_agents2*dB1[:,j].view(-1,1)) if j<=NT2 else y2_agents2).clamp(min=0,max=w)  ## y2_agents2*(1-y2_agents2/w)*
            y23_agents2=((y23_agents2+zy23_agents2*dB1[:,j].view(-1,1)) if j<=NT2 else y23_agents2).clamp(min=0,max=w)  ## y23_agents2*(1-y23_agents2/w)*
            y3_agents2=((y3_agents2+zy3_agents2*dB1[:,j].view(-1,1))).clamp(min=0,max=w)  ## y3_agents2*(1-y3_agents2/w)*


          S=(omiga1*((y1_agents1+y12_agents1+y13_agents1).mean())+\
            omiga2*((y1_agents2+y12_agents2+y13_agents2).mean())) * (j<NT1)+\
            (omiga1*((y2_agents1+y23_agents1).mean())+\
            omiga2*((y2_agents2+y23_agents2).mean()))*(j>=NT1 and j<NT2)+\
            (omiga1*((y3_agents1).mean())+\
            omiga2*((y3_agents2).mean())*(j>=NT2))

          g_agents1=((y1_agents1+y12_agents1+y13_agents1)*(j<NT1)+\
                    (y2_agents1+y23_agents1)*(j>=NT1 and j<NT2)+\
                    (y3_agents1)*(j>=NT2)
                    )/zeta1
          g_agents2=((y1_agents2+y12_agents2+y13_agents2)*(j<NT1)+\
                    (y2_agents2+y23_agents2)*(j>=NT1 and j<NT2)+\
                    (y3_agents2)*(j>=NT2)
                    )/zeta2

          Gamma_agents1 =((y1_agents1+y12_agents1+y13_agents1-S)*(j<NT1)+\
                          (y2_agents1+y23_agents1-S)*(j>=NT1 and j<NT2)+\
                          (y3_agents1-S)*(j>=NT2)
                          )/gamma1
          Gamma_agents2 =((y1_agents2+y12_agents2+y13_agents2-S)*(j<NT1)+\
                          (y2_agents2+y23_agents2-S)*(j>=NT1 and j<NT2)+\
                          (y3_agents2-S)*(j>=NT2)
                          )/gamma2
          
          a_agents1 =(((NT3-NT2)*y3_agents1 + (NT2-NT1)*(y2_agents1+y23_agents1) + (NT1-j)*(y1_agents1+y12_agents1+y13_agents1)) * (j<NT1)+\
                      ((NT3-NT2)*y3_agents1 + (NT2-j)*(y2_agents1+y23_agents1))*(j>=NT1 and j<NT2)+\
                      ((NT3-j)*y3_agents1)*(j>=NT2)
                      ) * dt/beta1
          a_agents2 =(((NT3-NT2)*y3_agents2 + (NT2-NT1)*(y2_agents2+y23_agents2) + (NT1-j)*(y1_agents2+y12_agents2+y13_agents2)) * (j<NT1)+\
                      ((NT3-NT2)*y3_agents2 + (NT2-j)*(y2_agents2+y23_agents2))*(j>=NT1 and j<NT2)+\
                      ((NT3-j)*y3_agents2)*(j>=NT2)
                      ) * dt/beta2

        ## -------------------------------- Record Paths -------------------------------- ##
        x_agents1_path[:,j] = x_agents1.squeeze()
        
        y1_agents1_path[:,j] = y1_agents1.squeeze()
        y12_agents1_path[:,j] = y12_agents1.squeeze()
        y13_agents1_path[:,j] = y13_agents1.squeeze()
        y2_agents1_path[:,j] = y2_agents1.squeeze()
        y23_agents1_path[:,j] = y23_agents1.squeeze()
        y3_agents1_path[:,j] = y3_agents1.squeeze()
       
        g_agents1_path[:,j] = g_agents1.squeeze()
        Gamma_agents1_path[:,j] = Gamma_agents1.squeeze()
        a_agents1_path[:,j] = a_agents1.squeeze()

        x_agents2_path[:,j] = x_agents2.squeeze()
        
        y1_agents2_path[:,j] = y1_agents2.squeeze()
        y12_agents2_path[:,j] = y12_agents2.squeeze()
        y13_agents2_path[:,j] = y13_agents2.squeeze()
        y2_agents2_path[:,j] = y2_agents2.squeeze()
        y23_agents2_path[:,j] = y23_agents2.squeeze()
        y3_agents2_path[:,j] = y3_agents2.squeeze()
       
        g_agents2_path[:,j] = g_agents2.squeeze()
        Gamma_agents2_path[:,j] = Gamma_agents2.squeeze()
        a_agents2_path[:,j] = a_agents2.squeeze()

        S_path[j]=S.squeeze()
        ## -------------------------------- ------------ -------------------------------- ##
        if j==NT1 or j==NT2:  # @ NT1 
          x_agents1 = nn.ReLU()(x_agents1-K)  # @ NT1: hand in min(K,xt1)
          x_agents2 = nn.ReLU()(x_agents2-K)
        
  cum_g_agents1_path=torch.zeros(size=(NumTrain,1),device=device)
  cum_g_agents1_path=torch.hstack([cum_g_agents1_path,dt*(g_agents1_path.cumsum(axis=1)[:,:-1])])
  cum_Gamma_agents1_path=torch.zeros(size=(NumTrain,1),device=device)
  cum_Gamma_agents1_path=torch.hstack([cum_Gamma_agents1_path,dt*(Gamma_agents1_path.cumsum(axis=1)[:,:-1])])
  cum_a_agents1_path=torch.zeros(size=(NumTrain,1),device=device)
  cum_a_agents1_path=torch.hstack([cum_a_agents1_path,dt*(a_agents1_path.cumsum(axis=1)[:,:-1])])

  cum_g_agents2_path=torch.zeros(size=(NumTrain,1),device=device)
  cum_g_agents2_path=torch.hstack([cum_g_agents2_path,dt*(g_agents2_path.cumsum(axis=1)[:,:-1])])
  cum_Gamma_agents2_path=torch.zeros(size=(NumTrain,1),device=device)
  cum_Gamma_agents2_path=torch.hstack([cum_Gamma_agents2_path,dt*(Gamma_agents2_path.cumsum(axis=1)[:,:-1])])
  cum_a_agents2_path=torch.zeros(size=(NumTrain,1),device=device)
  cum_a_agents2_path=torch.hstack([cum_a_agents2_path,dt*(a_agents2_path.cumsum(axis=1)[:,:-1])])
  
  base_agents1_path=torch.ones(NT3+1,device=device)*h1
  base_agents2_path=torch.ones(NT3+1,device=device)*h2
  cum_base_agents1_path=torch.linspace(0,h1,NT3+1,device=device)
  cum_base_agents2_path=torch.linspace(0,h2,NT3+1,device=device)

  pop1_path_dict={'inventory':x_agents1_path,
                  'price':S_path,
                  'expansion':a_agents1_path,
                  'generation':g_agents1_path,
                  'trading':Gamma_agents1_path,
                  'base':base_agents1_path,
                  'cum_expansion':cum_a_agents1_path,
                  'cum_generation':cum_g_agents1_path,
                  'cum_trading':cum_Gamma_agents1_path,
                  'cum_base':cum_base_agents1_path,
                  'y1':y1_agents1_path,
                  'y12':y12_agents1_path,
                  'y13':y13_agents1_path,
                  'y2':y2_agents1_path,
                  'y23':y23_agents1_path,
                  'y3':y3_agents1_path,
                }
  pop2_path_dict={'inventory':x_agents2_path,
                  'price':S_path,
                  'expansion':a_agents2_path,
                  'generation':g_agents2_path,
                  'trading':Gamma_agents2_path,
                  'base':base_agents2_path,
                  'cum_expansion':cum_a_agents2_path,
                  'cum_generation':cum_g_agents2_path,
                  'cum_trading':cum_Gamma_agents2_path,
                  'cum_base':cum_base_agents2_path,
                  'y1':y1_agents2_path,
                  'y12':y12_agents2_path,
                  'y13':y13_agents2_path,
                  'y2':y2_agents2_path,
                  'y23':y23_agents2_path,
                  'y3':y3_agents2_path,
                }
  
  pop1_path_dict = move_to_cpu(pop1_path_dict)
  pop2_path_dict = move_to_cpu(pop2_path_dict)

  return pop1_path_dict, pop2_path_dict

class plot_results():
    def __init__(self,pop1_dict, pop2_dict, loss, PlotPaths=100, seed=42, savefigs=False, to_path=None): #dB, init_x, init_c, GlobalParams, agents, loss,PlotPaths=100, seed=42):
        ## -------------------------------- Common Params -------------------------------- ##
        self.loss=torch.tensor(loss, device=torch.device('cpu'))
        self.target_type='indicator' if (pop1_dict['GlobalParams'].target_type=='indicator' and pop1_dict['GlobalParams'].trick!='logit') else "sigmoid"
        self.delta=pop1_dict['GlobalParams'].delta
        self.K=pop1_dict['GlobalParams'].K
        self.w=pop1_dict['GlobalParams'].w
        self.NT1=pop1_dict['GlobalParams'].NT1
        self.NT2=pop1_dict['GlobalParams'].NT2
        self.NT3=pop1_dict['GlobalParams'].NT3
        self.dt=pop1_dict['GlobalParams'].dt
        self.t = np.array([i for i in range(self.NT3+1)]) * self.dt
        self.NumTrain=pop1_dict['GlobalParams'].NumTrain
        self.number_of_paths=np.minimum(PlotPaths,self.NumTrain)
        self.seed=seed
        self.savefigs=savefigs
        self.to_path=to_path
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
        self.pop1_plot = {k:v[idx_list] for k,v in self.pop1_path_dict.items() if v.shape[0]==self.NumTrain}
        self.pop2_plot = {k:v[idx_list] for k,v in self.pop2_path_dict.items() if v.shape[0]==self.NumTrain}
        self.pop1_plot.update({k:v for k,v in self.pop1_path_dict.items() if v.shape[0]!=self.NumTrain})
        self.pop2_plot.update({k:v for k,v in self.pop2_path_dict.items() if v.shape[0]!=self.NumTrain})
    
    def FwdLoss(self,log=True):
        plt.figure(figsize=(8,5))
        plt.title("Forward_Loss vs Epoch",fontsize=18)
        plt.plot(self.loss,color='firebrick',alpha=0.6)
        if log==True:
            plt.yscale('log')
        if self.savefigs and self.to_path:
            plt.savefig(self.to_path.joinpath("Forward_Loss.png"),bbox_inches='tight')

    def Inventory_And_Price(self,Histogram=True): #,single_sample=False):
        plt.figure(figsize=(14,6))
        plt.subplot(121)
        plt.title("$Inventory(X_t)$")
        for i in range(self.number_of_paths):
          ax1=plt.plot(self.t[:self.NT1+1],self.pop1_plot['inventory'][i][:self.NT1+1], 
                       self.t[self.NT1+1:self.NT2+1],self.pop1_plot['inventory'][i][self.NT1+1:self.NT2+1],
                       self.t[self.NT2+1:],self.pop1_plot['inventory'][i][self.NT2+1:],
                       color="green", linewidth=1, alpha=0.3)
          ax2=plt.plot(self.t[:self.NT1+1],self.pop2_plot['inventory'][i][:self.NT1+1], 
                       self.t[self.NT1+1:self.NT2+1],self.pop2_plot['inventory'][i][self.NT1+1:self.NT2+1],
                       self.t[self.NT2+1:],self.pop2_plot['inventory'][i][self.NT2+1:],
                       color="firebrick", linewidth=1, alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(122)
        plt.title("$Price(S_t)$")
        plt.plot(self.t, self.pop1_plot['price'],color='darkgrey')
        # plt.ylim(-0.1,1.5)
        if self.savefigs and self.to_path:
           plt.savefig(self.to_path.joinpath("Inventory-Price.png"),bbox_inches='tight')

        if Histogram==True:
          x1_t1=self.pop1_path_dict['inventory'][:,self.NT1]
          x2_t1=self.pop2_path_dict['inventory'][:,self.NT1]
          x1_t2=self.pop1_path_dict['inventory'][:,self.NT2]

          x2_t2=self.pop2_path_dict['inventory'][:,self.NT2]
          x1_t3=self.pop1_path_dict['inventory'][:,self.NT3]
          x2_t3=self.pop2_path_dict['inventory'][:,self.NT3]
          ## -------------------------------- Population 1 -------------------------------- ##
          plt.figure(figsize=(14,4))
          plt.suptitle("Inventory @ Delivery - P1")

          plt.subplot(131)
          plt.title("Distribution of $X_{T_1}^{(1)}$")
          sns.histplot(data=x1_t1, bins=100,stat='count',alpha=0.6,color= 'green')
          sns.kdeplot(x1_t1, color="green",label='P1')
          plt.xlabel("$X_{T_1}^{(1)}$")
          plt.ylabel("Count")
          plt.legend()

          plt.subplot(132)
          plt.title("Distribution of $X_{T_2}^{(1)}$")
          sns.histplot(data=x1_t2, bins=100,stat='count',alpha=0.6,color= 'green')
          sns.kdeplot(x1_t2, color="green",label='P1')
          plt.xlabel("$X_{T_2}^{(1)}$")
          plt.ylabel("Count")
          plt.legend()

          plt.subplot(133)
          plt.title("Distribution of $X_{T_3}^{(1)}$")
          sns.histplot(data=x1_t3, bins=100,stat='count',alpha=0.6,color= 'green')
          sns.kdeplot(x1_t3, color="green",label='P1')
          plt.xlabel("$X_{T_3}^{(1)}$")
          plt.ylabel("Count")
          plt.legend()
          if self.savefigs and self.to_path:
            plt.savefig(self.to_path.joinpath("Inventory-Distribution-pop1.png"),bbox_inches='tight')
          ## -------------------------------- Population 2 -------------------------------- ##
          plt.figure(figsize=(14,4))
          plt.suptitle("Inventory @ Delivery - P2")

          plt.subplot(131)
          plt.title("Distribution of $X_{T_1}^{(2)}$")
          sns.histplot(data=x2_t1, bins=100,stat='count',alpha=0.6,color= 'firebrick')
          sns.kdeplot(x2_t1, color="firebrick",label='P2')
          plt.xlabel("$X_{T_1}^{(2)}$")
          plt.ylabel("Count")
          plt.legend()

          plt.subplot(132)
          plt.title("Distribution of $X_{T_2}^{(2)}$")
          sns.histplot(data=x2_t2, bins=100,stat='count',alpha=0.6,color= 'firebrick')
          sns.kdeplot(x2_t2, color="firebrick",label='P2')
          plt.xlabel("$X_{T_2}^{(2)}$")
          plt.ylabel("Count")
          plt.legend()

          plt.subplot(133)
          plt.title("Distribution of $X_{T_3}^{(2)}$")
          sns.histplot(data=x2_t3, bins=100,stat='count',alpha=0.6,color= 'firebrick')
          sns.kdeplot(x2_t3, color="firebrick",label='P2')
          plt.xlabel("$X_{T_3}^{(2)}$")
          plt.ylabel("Count")
          plt.legend()

          if self.savefigs and self.to_path:
            plt.savefig(self.to_path.joinpath("Inventory-Distribution-pop2.png"),bbox_inches='tight')

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
          ax1=plt.plot(self.t,self.pop1_plot['expansion'][i], color="green", linewidth=1, alpha=0.3)
          ax2=plt.plot(self.t,self.pop2_plot['expansion'][i], color="firebrick", linewidth=1, alpha=0.3)
        # plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(132 if base_rate==False else 143)
        plt.title("Generation Rate")
        for i in range(self.number_of_paths):
          ax1=plt.plot(self.t,self.pop1_plot['generation'][i], color="green", linewidth=1, alpha=0.3)
          ax2=plt.plot(self.t,self.pop2_plot['generation'][i], color="firebrick", linewidth=1, alpha=0.3)
        # plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(133 if base_rate==False else 144)
        plt.title("Trading Rate")
        for i in range(self.number_of_paths):
          ax1=plt.plot(self.t,self.pop1_plot['trading'][i], color="green", linewidth=1, alpha=0.3)
          ax2=plt.plot(self.t,self.pop2_plot['trading'][i], color="firebrick", linewidth=1, alpha=0.3)
        # plt.ylim(-0.5,1)
        plt.legend({'P1':ax1,'P2':ax2})
        if self.savefigs and self.to_path:
          plt.savefig(self.to_path.joinpath("Rates.png"),bbox_inches='tight')

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
            plt.title("Accumulated Expansion")
            for i in range(self.number_of_paths):
              ax1=plt.plot(self.t,self.pop1_plot['cum_expansion'][i], color="green", linewidth=1, alpha=0.3)
              ax2=plt.plot(self.t,self.pop2_plot['cum_expansion'][i], color="firebrick", linewidth=1, alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(132 if base_rate==False else 143)
            plt.title("Accumulated Generation")
            for i in range(self.number_of_paths):
              ax1=plt.plot(self.t,self.pop1_plot['cum_generation'][i], color="green", linewidth=1, alpha=0.3)
              ax2=plt.plot(self.t,self.pop2_plot['cum_generation'][i], color="firebrick", linewidth=1, alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})

            plt.subplot(133 if base_rate==False else 144)
            plt.title("Accumulated Trading")
            for i in range(self.number_of_paths):
              ax1=plt.plot(self.t,self.pop1_plot['cum_trading'][i], color="green", linewidth=1, alpha=0.3)
              ax2=plt.plot(self.t,self.pop2_plot['cum_trading'][i], color="firebrick", linewidth=1, alpha=0.3)
            plt.legend({'P1':ax1,'P2':ax2})
            if self.savefigs and self.to_path:
              plt.savefig(self.to_path.joinpath("AccumRates.png"),bbox_inches='tight')

    def Key_Processes(self):
        plt.figure(figsize=(14,14))

        plt.subplot(331)
        plt.title("$Y1_t$")
        for i in range(self.number_of_paths):
            ax1=plt.plot(self.t,self.pop1_path_dict['y1'][i], color="green", linewidth=1, alpha=0.3)
            ax2=plt.plot(self.t,self.pop2_path_dict['y1'][i], color="firebrick", linewidth=1, alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(332)
        plt.title("$Y12_t$")
        for i in range(self.number_of_paths):
            ax1=plt.plot(self.t,self.pop1_path_dict['y12'][i], color="green", linewidth=1, alpha=0.3)
            ax2=plt.plot(self.t,self.pop2_path_dict['y12'][i], color="firebrick", linewidth=1, alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(333)
        plt.title("$Y13_t$")
        for i in range(self.number_of_paths):
            ax1=plt.plot(self.t,self.pop1_path_dict['y13'][i], color="green", linewidth=1, alpha=0.3)
            ax2=plt.plot(self.t,self.pop2_path_dict['y13'][i], color="firebrick", linewidth=1, alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})
        
        plt.subplot(334)
        plt.title("$Y2_t$")
        for i in range(self.number_of_paths):
            ax1=plt.plot(self.t,self.pop1_path_dict['y2'][i], color="green", linewidth=1, alpha=0.3)
            ax2=plt.plot(self.t,self.pop2_path_dict['y2'][i], color="firebrick", linewidth=1, alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})
        
        plt.subplot(335)
        plt.title("$Y23_t$")
        for i in range(self.number_of_paths):
            ax1=plt.plot(self.t,self.pop1_path_dict['y23'][i], color="green", linewidth=1, alpha=0.3)
            ax2=plt.plot(self.t,self.pop2_path_dict['y23'][i], color="firebrick", linewidth=1, alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})

        plt.subplot(337)
        plt.title("$Y3_t$")
        for i in range(self.number_of_paths):
            ax1=plt.plot(self.t,self.pop1_path_dict['y3'][i], color="green", linewidth=1, alpha=0.3)
            ax2=plt.plot(self.t,self.pop2_path_dict['y3'][i], color="firebrick", linewidth=1, alpha=0.3)
        plt.legend({'P1':ax1,'P2':ax2})

        if self.savefigs and self.to_path:
            plt.savefig(self.to_path.joinpath("Key_Processes.png"),bbox_inches='tight')

    def Terminal_Convergence(self,Fitted=True):
        str_x_t1_agents1, str_x_t2_agents1, str_x_t3_agents1 = ("$X^{(1)}_{T_1}$", "$X^{(1)}_{T_2}$", "$X^{(1)}_{T_3}$")
        str_y1_t1_agents1, str_y12_t1_agents1, str_y13_t1_agents1 = ("$Y1^{(1)}_{T_1}$", "$Y12^{(1)}_{T_1}$", "$Y13^{(1)}_{T_1}$")
        str_y2_t2_agents1, str_y23_t2_agents1 = ("$Y2^{(1)}_{T_1}$", "$Y23^{(1)}_{T_1}$", "$Y2^{(1)}_{T_2}$", "$Y23^{(1)}_{T_2}$")
        str_y3_t3_agents1 = ("$Y3^{(1)}_{T_3}$")
        
        str_x_t1_agents2, str_x_t2_agents2, str_x_t3_agents2 = ("$X^{(2)}_{T_1}$", "$X^{(2)}_{T_2}$", "$X^{(2)}_{T_3}$")
        str_y1_t1_agents2, str_y12_t1_agents2, str_y13_t1_agents2 = ("$Y1^{(2)}_{T_1}$", "$Y12^{(2)}_{T_1}$", "$Y13^{(2)}_{T_1}$")
        str_y2_t2_agents2, str_y23_t2_agents2 = ("$Y2^{(2)}_{T_1}$", "$Y23^{(2)}_{T_1}$", "$Y2^{(2)}_{T_2}$", "$Y23^{(2)}_{T_2}$")
        str_y3_t3_agents2 = ("$Y3^{(2)}_{T_3}$")

        ## -------------------------------- Targets -------------------------------- ##
        x_t1_agents1=self.pop1_path_dict['inventory'][:,self.NT1]
        x_t2_agents1=self.pop1_path_dict['inventory'][:,self.NT2]
        x_t3_agents1=self.pop1_path_dict['inventory'][:,self.NT3]

        x_t2_agents2=self.pop1_path_dict['inventory'][:,self.NT2]
        x_t3_agents2=self.pop1_path_dict['inventory'][:,self.NT3]
        x_t1_agents2=self.pop2_path_dict['inventory'][:,self.NT1]  ##[0,NT1] --> init @ NT1 --> [NT1,Nt2]
        
        target_y1_agents1=self.w*target(x_end=x_t1_agents1,GlobalParams=self.GlobalParams1,target_type=self.target_type, device='cpu')
        target_y2_agents1=self.w*target(x_end=x_t2_agents1,GlobalParams=self.GlobalParams1,target_type=self.target_type, device='cpu')
        target_y3_agents1=self.w*target(x_end=x_t3_agents1,GlobalParams=self.GlobalParams1,target_type=self.target_type, device='cpu')
        target_y12_agents1 = (1-target_y1_agents1/self.w)*target_y2_agents1
        target_y13_agents1 = (1-target_y1_agents1/self.w)*target_y3_agents1
        target_y23_agents1 = (1-target_y2_agents1/self.w)*target_y3_agents1

        target_y1_agents2=self.w*target(x_end=x_t1_agents2,GlobalParams=self.GlobalParams1,target_type=self.target_type, device='cpu')
        target_y2_agents2=self.w*target(x_end=x_t2_agents2,GlobalParams=self.GlobalParams1,target_type=self.target_type, device='cpu')
        target_y3_agents2=self.w*target(x_end=x_t3_agents2,GlobalParams=self.GlobalParams1,target_type=self.target_type, device='cpu')
        target_y12_agents2 = (1-target_y1_agents2/self.w)*target_y2_agents2
        target_y13_agents2 = (1-target_y1_agents2/self.w)*target_y3_agents2
        target_y23_agents2 = (1-target_y2_agents2/self.w)*target_y3_agents2

        if Fitted==True:
          ## -------------------------------- Population 1 -------------------------------- ##
          fig=plt.figure(figsize=(14,14))
          plt.suptitle(f"Termianl Values - Pop1")

          ax331=fig.add_subplot(331)
          ax331.set_title(str_y1_t1_agents1)
          ax1=ax331.scatter(x_t1_agents1,target_y1_agents1,s=1,alpha=0.3,color='black')
          ax2=ax331.scatter(x_t1_agents1,self.pop1_path_dict['y1'][:,self.NT1],s=1,alpha=0.5,color='green')
          ax331.set_xlabel(str_x_t1_agents1)
          ax331.set_ylabel(str_y1_t1_agents1) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t1_agents1}<0.9)':ax1,
                             f'Learnt: {str_y1_t1_agents1}':ax2
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 


          ax332=fig.add_subplot(332, projection='3d')
          ax332.set_title(str_y12_t1_agents1)
          x, y = np.meshgrid(x_t1_agents1, x_t2_agents1)
          ax1=ax332.scatter(x,y,target_y12_agents1,s=1,alpha=0.3,color='black')
          ax2=ax332.scatter(x,y,self.pop1_path_dict['y12'][:,self.NT1],s=1,alpha=0.5,color='green')
          ax332.set_xlabel(str_x_t1_agents1)
          ax332.set_ylabel(str_x_t2_agents1)
          ax332.set_zlabel(str_y12_t1_agents1) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t1_agents1}>=0.9)*{self.target_type}({str_x_t2_agents1}<0.9)':ax1,
                             f'Learnt: {str_y12_t1_agents1}':ax2,
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0])
          
          ax333=fig.add_subplot(333, projection='3d')
          ax333.set_title(str_y13_t1_agents1)
          x, y = np.meshgrid(x_t1_agents1, x_t3_agents1)
          ax1=ax333.scatter(x,y,target_y13_agents1,s=1,alpha=0.3,color='black')
          ax2=ax333.scatter(x,y,self.pop1_path_dict['y13'][:,self.NT1],s=1,alpha=0.5,color='green')
          ax333.set_xlabel(str_x_t1_agents1)
          ax333.set_ylabel(str_x_t3_agents1)
          ax333.set_zlabel(str_y13_t1_agents1) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t1_agents1}>=0.9)*{self.target_type}({str_x_t3_agents1}<0.9)':ax1,
                             f'Learnt: {str_y13_t1_agents1}':ax2,
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0])

          ax334=fig.add_subplot(334)
          ax334.set_title(str_y2_t2_agents1)
          ax1=ax334.scatter(x_t2_agents1,target_y2_agents1,s=1,alpha=0.3,color='black')
          ax2=ax334.scatter(x_t2_agents1,self.pop1_path_dict['y2'][:,self.NT2],s=1,alpha=0.5,color='green')
          ax334.set_xlabel(str_x_t2_agents1)
          ax334.set_ylabel(str_y2_t2_agents1) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t2_agents1}<0.9)':ax1,
                             f'Learnt: {str_y2_t2_agents1}':ax2
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 

          ax335=fig.add_subplot(335, projection='3d')
          ax335.set_title(str_y23_t2_agents1)
          x, y = np.meshgrid(x_t2_agents1, x_t3_agents1)
          ax1=ax335.scatter(x,y,target_y23_agents1,s=1,alpha=0.3,color='black')
          ax2=ax335.scatter(x,y,self.pop1_path_dict['y23'][:,self.NT2],s=1,alpha=0.5,color='green')
          ax335.set_xlabel(str_x_t2_agents1)
          ax335.set_ylabel(str_x_t3_agents1)
          ax335.set_zlabel(str_y23_t2_agents1)
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t2_agents1}>=0.9)*{self.target_type}({str_x_t3_agents1}<0.9)':ax1,
                             f'Learnt: {str_y23_t2_agents1}':ax2,
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0])
          
          ax337=fig.add_subplot(337)
          ax337.set_title(str_y3_t3_agents1)
          ax1=ax337.scatter(x_t3_agents1,target_y3_agents1,s=1,alpha=0.3,color='black')
          ax2=ax337.scatter(x_t3_agents1,self.pop1_path_dict['y3'][:,self.NT2],s=1,alpha=0.5,color='green')
          ax337.set_xlabel(str_x_t3_agents1)
          ax337.set_ylabel(str_y3_t3_agents1)
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t3_agents1}<0.9)':ax1,
                             f'Learnt: {str_y3_t3_agents1}':ax2
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 

          ## -------------------------------- Population 2 -------------------------------- ##
          fig=plt.figure(figsize=(14,14))
          plt.suptitle(f"Termianl Values - Pop1")

          ax331=fig.add_subplot(331)
          ax331.set_title(str_y1_t1_agents2)
          ax1=ax331.scatter(x_t1_agents2,target_y1_agents2,s=1,alpha=0.3,color='black')
          ax2=ax331.scatter(x_t1_agents2,self.pop2_path_dict['y1'][:,self.NT1],s=1,alpha=0.5,color='green')
          ax331.set_xlabel(str_x_t1_agents2)
          ax331.set_ylabel(str_y1_t1_agents2) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t1_agents2}<0.9)':ax1,
                             f'Learnt: {str_y1_t1_agents2}':ax2
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 


          ax332=fig.add_subplot(332, projection='3d')
          ax332.set_title(str_y12_t1_agents2)
          x, y = np.meshgrid(x_t1_agents2, x_t2_agents2)
          ax1=ax332.scatter(x,y,target_y12_agents2,s=1,alpha=0.3,color='black')
          ax2=ax332.scatter(x,y,self.pop2_path_dict['y12'][:,self.NT1],s=1,alpha=0.5,color='green')
          ax332.set_xlabel(str_x_t1_agents2)
          ax332.set_ylabel(str_x_t2_agents2)
          ax332.set_zlabel(str_y12_t1_agents2) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t1_agents2}>=0.9)*{self.target_type}({str_x_t2_agents2}<0.9)':ax1,
                             f'Learnt: {str_y12_t1_agents2}':ax2,
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0])
          
          ax333=fig.add_subplot(333, projection='3d')
          ax333.set_title(str_y13_t1_agents2)
          x, y = np.meshgrid(x_t1_agents2, x_t3_agents2)
          ax1=ax333.scatter(x,y,target_y13_agents2,s=1,alpha=0.3,color='black')
          ax2=ax333.scatter(x,y,self.pop2_path_dict['y13'][:,self.NT1],s=1,alpha=0.5,color='green')
          ax333.set_xlabel(str_x_t1_agents2)
          ax333.set_ylabel(str_x_t3_agents2)
          ax333.set_zlabel(str_y13_t1_agents2) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t1_agents2}>=0.9)*{self.target_type}({str_x_t3_agents2}<0.9)':ax1,
                             f'Learnt: {str_y13_t1_agents2}':ax2,
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0])

          ax334=fig.add_subplot(334)
          ax334.set_title(str_y2_t2_agents2)
          ax1=ax334.scatter(x_t2_agents2,target_y2_agents2,s=1,alpha=0.3,color='black')
          ax2=ax334.scatter(x_t2_agents2,self.pop2_path_dict['y2'][:,self.NT2],s=1,alpha=0.5,color='green')
          ax334.set_xlabel(str_x_t2_agents2)
          ax334.set_ylabel(str_y2_t2_agents2) 
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t2_agents2}<0.9)':ax1,
                             f'Learnt: {str_y2_t2_agents2}':ax2
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 

          ax335=fig.add_subplot(335, projection='3d')
          ax335.set_title(str_y23_t2_agents2)
          x, y = np.meshgrid(x_t2_agents2, x_t3_agents2)
          ax1=ax335.scatter(x,y,target_y23_agents2,s=1,alpha=0.3,color='black')
          ax2=ax335.scatter(x,y,self.pop2_path_dict['y23'][:,self.NT2],s=1,alpha=0.5,color='green')
          ax335.set_xlabel(str_x_t2_agents2)
          ax335.set_ylabel(str_x_t3_agents2)
          ax335.set_zlabel(str_y23_t2_agents2)
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t2_agents2}>=0.9)*{self.target_type}({str_x_t3_agents2}<0.9)':ax1,
                             f'Learnt: {str_y23_t2_agents2}':ax2,
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0])
          
          ax337=fig.add_subplot(337)
          ax337.set_title(str_y3_t3_agents2)
          ax1=ax337.scatter(x_t3_agents2,target_y3_agents2,s=1,alpha=0.3,color='black')
          ax2=ax337.scatter(x_t3_agents2,self.pop2_path_dict['y3'][:,self.NT2],s=1,alpha=0.5,color='green')
          ax337.set_xlabel(str_x_t3_agents2)
          ax337.set_ylabel(str_y3_t3_agents2)
          legend=plt.legend({f'Target: {self.w}*{self.target_type}({str_x_t3_agents2}<0.9)':ax1,
                             f'Learnt: {str_y3_t3_agents2}':ax2
                            }, 
                            fontsize=8,framealpha=0.6, bbox_to_anchor=(0.8, -0.15))
          for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1)  
            text.set_color(handle.get_facecolor()[0]) 
