import os
import numpy as np
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt
import scipy.signal as scisi
import librosa
import librosa.core
import librosa.feature
import sys
import librosa.display
import numpy.matlib
from sklearn import metrics
import sklearn.svm as svm
import sklearn.ensemble as sken
import pywt
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxopt import matrix, solvers  
import sklearn.metrics.pairwise as smp
import pickle
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
import time
path_to_folder = "C:/Users/frusque/Spyder_Data/Focus_SVDD_EAD/"
path=path_to_folder +"MC_UD_3_09 5,10,15cm/MC_UD_3_09 All data/"
ld=np.sort(os.listdir(path))
ld=np.concatenate((ld[3:],ld[0:3]))
import scipy.io

DataF=[]
DataT=[]

for i in range(len(ld)):
    print([i,ld[i]])
    M=scipy.io.loadmat(path+ld[i])
    print(np.shape(M["A"])[1])
    for j in range(np.shape(M["A"])[1]):
        c=M["A"][:,j]
        #c=c-np.mean(c)
        c=np.array(c,dtype=np.double)
        if i <6:
            DataF.append(c)
        else:
            DataT.append(c)
            
#Nmax=np.max(DataT) 
DataT=np.array(DataT)
DataF=np.array(DataF)
a=np.mean(DataT)
b=np.std(DataT)
DataT=(DataT-a)/b
DataF=(np.array(DataF)-a)/b

import sys
sys.path.insert(1, path_to_folder)
from focus_SVDD import  Generation_data, AEnet_c, AEnet, GetF1Score
MSE=nn.MSELoss()
def train(network,kind):
    network.train()
    loss_v=[]
    if kind=="Hilbert":
        for i in range(10):
            optimizer.zero_grad()
            Input=GG.__getitem__(BS=BS)
            Output = network(Input)
            loss=MSE(torch.abs(Output-Input), torch.zeros_like(Input,dtype=torch.float))
            loss.backward()
            optimizer.step() 
            loss_v=loss.item()
    else:
        for i in range(10):
            optimizer.zero_grad()
            Input=GG.__getitem__(BS=BS)
            Output = network(Input)
            loss=MSE(Output,Input)
            loss.backward()
            optimizer.step() 
            loss_v=loss.item()
    return np.mean(loss_v)



kind="Hilbert"#Option are Real, Amp, H_amp_phi, H_real_imag, LFFT
Activation="EAD"#Option CReLU, modReLU (if signal is real then ReLU)

BS=32#Batch size
N_Epoch=2000#number of Epoch
n_cross=5#Nb of cross validation
path_result=path_to_folder+"Results/Scores_seed_"


F1_withoutSVDD=[]
F1_withSVDD=[]
for seed in range(0,n_cross,1):
    GG=Generation_data(DataT,DataF,42) 
    GG.Mix(k=seed)
    GG.Get_X(kind)
    if kind=="Hilbert":
        r=GG.Test(Train=np.abs(GG.XTrain),Valid=np.abs(GG.XValid),Test=np.abs(GG.XTest))
        F1_withoutSVDD.append(GetF1Score(r))
        network=AEnet_c(In=1501,fun=Activation).double().to(DEVICE)
        optimizer=torch.optim.Adam(network.parameters(), lr=0.001)
        scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,800,1200,1600], gamma=0.2)
        for epoch in range(N_Epoch):
            l=train(network,kind)
            #print([seed,epoch,l])
            scheduler.step() 
        r=GG.Test(Train=np.abs(GG.XTrain-GG.gn_(network(GG.gt_(GG.XTrain)))),
                Valid=np.abs(GG.XValid-GG.gn_(network(GG.gt_(GG.XValid)))),
                Test=np.abs(GG.XTest-GG.gn_(network(GG.gt_(GG.XTest)))))
        F1_withSVDD.append(GetF1Score(r))
    else:
        r=GG.Test(Train=GG.XTrain,Valid=GG.XValid,Test=GG.XTest)
        network=AEnet(In=1501).double().to(DEVICE)
        F1_withoutSVDD.append(GetF1Score(r))
        optimizer=torch.optim.Adam(network.parameters(), lr=0.001)
        scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,800,1200,1600], gamma=0.5)
        for epoch in range(N_Epoch):
            l=train(network,kind)
            #print([seed,epoch,l])
            scheduler.step() 
        r=GG.Test(Train=GG.XTrain-GG.gn_(network(GG.gt_(GG.XTrain))),
                Valid=GG.XValid-GG.gn_(network(GG.gt_(GG.XValid))),
                Test=GG.XTest-GG.gn_(network(GG.gt_(GG.XTest))))
        F1_withSVDD.append(GetF1Score(r))

print(np.mean(F1_withSVDD,0))
print(np.mean(F1_withoutSVDD,0))
#%%

0.9346879535558782
0.9661705006765899
0.9530386740331491