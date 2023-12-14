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
        c=np.array(c,dtype=np.double)
        if i <6:
            DataF.append(c)
        else:
            DataT.append(c)
            

DataT=np.array(DataT)
a=np.mean(DataT)
b=np.std(DataT)
DataT=(DataT-a)/b
DataF=(np.array(DataF)-a)/b
#%%
import sys
sys.path.insert(1, path_to_folder)
from focus_SVDD import  Generation_data, AEnet_c, AEnet
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


Kind_type=["Real"] #
InT=["AE_1"]
mod=""
BS=32
N_Epoch=2000
n_seed=5
N_cas=71
GG=Generation_data(DataT,DataF,seed=42)#Generator containing the Training, Validation, Test dataset
path_result=path_to_folder+"Results/Scores_seed_"
Kind_type=["Real","Hilbert"]
for seed in range(0,n_seed,1):
    GG=Generation_data(DataT,DataF,seed=42) 
    GG.Mix(k=seed)
    result={}
    for kind in Kind_type:
        GG.Get_X(kind)
        InB=[[len(GG.XTrain[0]),128,128,32]]
        # InB=[[len(GG.XTrain[0]),64,64,32],
        #       [len(GG.XTrain[0]),128,128,32]]
        r=GG.Test(Train=GG.XTrain,Valid=GG.XValid,Test=GG.XTest)
        Name=str(seed)+"_"+kind+"_Raw"
        result[Name]=r
        print(kind)
        if kind=="Hilbert":
            for IndexAE in range(len(InB)):
                for i_fun in range(3):
                    print(i_fun)
                    network=AEnet_c(In=InB[IndexAE],fun=i_fun+1).double().to(DEVICE)
                    optimizer=torch.optim.Adam(network.parameters(), lr=0.001)
                    scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,800,1200,1600], gamma=0.5)
                    for epoch in range(N_Epoch):
                        l=train(network,kind)
                        print([epoch,l])
                        scheduler.step() 
                    r=GG.Test(Train=np.abs(GG.XTrain-GG.gn_(network(GG.gt_(GG.XTrain)))),
                           Valid=np.abs(GG.XValid-GG.gn_(network(GG.gt_(GG.XValid)))),
                           Test=np.abs(GG.XTest-GG.gn_(network(GG.gt_(GG.XTest)))))
                    Name=str(seed)+"_"+kind+"_"+InT[IndexAE]+"_"+str(i_fun)+"_Rec"
                    result[Name]=r
                    r=GG.Test(Train=np.abs(GG.gn_(network.embed(GG.gt_(GG.XTrain)))),
                           Valid=np.abs(GG.gn_(network.embed(GG.gt_(GG.XValid)))),
                           Test=np.abs(GG.gn_(network.embed(GG.gt_(GG.XTest)))))
                    Name=str(seed)+"_"+kind+"_"+InT[IndexAE]+"_"+str(i_fun)+"_Min"
                    result[Name]=r
        else:
            for IndexAE in range(len(InB)):
                network=AEnet(In=InB[IndexAE]).double().to(DEVICE)
                optimizer=torch.optim.Adam(network.parameters(), lr=0.001)
                scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,800,1200,1600], gamma=0.5)
                for epoch in range(N_Epoch):
                    l=train(network,kind)
                    print([epoch,l])
                    scheduler.step() 
                r=GG.Test(Train=GG.XTrain-GG.gn_(network(GG.gt_(GG.XTrain))),
                        Valid=GG.XValid-GG.gn_(network(GG.gt_(GG.XValid))),
                        Test=GG.XTest-GG.gn_(network(GG.gt_(GG.XTest))))
                Name=str(seed)+"_"+kind+"_"+InT[IndexAE]+"_Rec"
                result[Name]=r
                r=GG.Test(Train=GG.gn_(network.embed(GG.gt_(GG.XTrain))),
                        Valid=GG.gn_(network.embed(GG.gt_(GG.XValid))),
                        Test=GG.gn_(network.embed(GG.gt_(GG.XTest))))
                Name=str(seed)+"_"+kind+"_"+InT[IndexAE]+"_Min"
                result[Name]=r
    with open(path_result+str(seed)+".p", 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        


import pickle 
res={}
sel=3 #Which score you want to look at 3 for F1score
nseed=5
for i in range(nseed):

    with open(path_result+str(i)+".p", 'rb') as handle:
        b = pickle.load(handle)
        
        
    keylist=list(b.keys())
    
    
    for key in keylist:
        M=b[key]
        r1=M[0][1][sel]
        M=M[1:]
        r2=[0,0,0,0]
        for style in range(4):
            c=np.array(M)[:,style,0]
            c[c==0]=1
            c=np.abs(c-0.05)
            cm=np.min(c)
            idx=np.max(np.where(c==cm))
            r2[style]=np.array(M)[idx,style,sel]
            #r2[style]=np.max(1-np.array(M)[:,style,sel])
        if key[2:]=="Real_R_Rec":
            print(r2)
        if i>=1:
            res[key[2:]]=np.concatenate([[r1], r2 ])/nseed+res[key[2:]]
        else:
            res[key[2:]]=np.concatenate([[r1], r2 ])/nseed
print(res)