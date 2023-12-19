import numpy as np
import numpy.matlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxopt import matrix, solvers  
import sklearn.metrics.pairwise as smp
import pickle
DEVICE = torch.device("cpu" ) 
import time
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import scipy

def TestBatterie(Y,T,s):
    a=accuracy_score(T,Y)
    b=roc_auc_score(T,s)
    c=f1_score(T,Y)
    d=precision_score(T,Y)
    e=recall_score(T,Y)
    return [a,b,c,d,e]
    


class Generation_data:
    def __init__(self, DataT, DataF, seed = 0):
        """ 
        DESCRIPTION
        
        Data Generator to creat Train, Valid and Test dataset
    
        --------------------------------------------------        
        INPUT
        healthy data        DataT (n*d) 
                        n: number of samples
                        d: number of features
        abnormal data        DataF (n*d) 
                        n: number of samples
                        d: number of features
            
        --------------------------------------------------
        
        """    
        
        self.VecAno = np.concatenate((3*np.ones((60+59,)),
                                 4*np.ones((61+62,)),
                                 5*np.ones((61+59,))))
        self.VecHlt = np.concatenate((0*np.ones((185,)),
                                 1*np.ones((196,)),
                                 2*np.ones((149,))))
        self.DataT=DataT
        self.DataF=DataF
        self.XTrain=[]
        self.XTest=[]
        self.XValid=[]
        self.lF=len(DataF)
        self.lT=len(DataT)
        self.of=np.arange(self.lF)
        
        self.Kind=[]
        a=[1,2,3,4,5,6,7,8,9,10]
        self.Lambda =np.concatenate([np.array(a)*10**(n) for n in range(-6,4,1)])
        np.random.seed(seed)
        self.ot=np.arange(self.lT)
        np.random.shuffle(self.ot)
        
    def Mix(self, k=0, mix=[330,100,100]):
        """ 
        DESCRIPTION
        
        Create the Train, Valid and Test dataset
        
        
        --------------------------------------------------        
        INPUT
        k       seed

        --------------------------------------------------
        
        """
        
        Id_f = self.ot[np.arange(k*100,(k+1)*100)] 
        Id_ = np.setdiff1d(self.ot,Id_f)
        Id_v = np.random.choice(Id_,mix[1],replace=False)
        Id_t = np.setdiff1d(Id_,Id_v)

        self.DataTrain = self.DataT[Id_t]
        self.DataTrain=self.DataTrain
        self.OCTrain = self.VecHlt[Id_t]
        self.DataValid = self.DataT[Id_v]
        self.OCValid = self.VecHlt[Id_v]
        self.DataTest = np.concatenate((self.DataT[Id_f],self.DataF),axis=0)
        self.LabelTest = np.concatenate((np.zeros((mix[2],)),np.ones((self.lF,))))
        self.OCTest = np.concatenate((self.VecHlt[Id_f],self.VecAno))
        self.DataT=[]
        self.DataF=[]
        
    def Get_X(self, Kind="Real"):
        """ 
        DESCRIPTION
        
        Use the selected representation for the signals
        
        
        --------------------------------------------------        
        INPUT
        Kind       "Real", "Hilber", "amp" (intantaneous envelope), 
                   "H_amp_phi" (concatenated instantaneous envelope and phase)
                   "H_real_imag" (concatenated real and imaginary part)
                   "LFFT" (log spectrum)

        --------------------------------------------------
        
        """
        self.Kind=Kind
        if Kind == "Real":
            self.XTrain=np.array(self.DataTrain)
            self.XValid=np.array(self.DataValid)
            self.XTest=np.array(self.DataTest)
            self.gauss="gauss"
        elif Kind == "Hilbert":
            self.XTrain=scipy.signal.hilbert(np.array(self.DataTrain))
            self.XValid=scipy.signal.hilbert(np.array(self.DataValid))
            self.XTest=scipy.signal.hilbert(np.array(self.DataTest))  
            self.gauss="gauss"
        elif Kind == "Amp":
            self.XTrain=np.abs(scipy.signal.hilbert(np.array(self.DataTrain)))
            self.XValid=np.abs(scipy.signal.hilbert(np.array(self.DataValid)))
            self.XTest=np.abs(scipy.signal.hilbert(np.array(self.DataTest))) 
            self.gauss="gauss"
        elif Kind == "H_amp_phi":
            self.XTrain=np.concatenate([np.abs(scipy.signal.hilbert(np.array(self.DataTrain))),np.angle(scipy.signal.hilbert(np.array(self.DataTrain)))],1)
            self.XValid=np.concatenate([np.abs(scipy.signal.hilbert(np.array(self.DataValid))),np.angle(scipy.signal.hilbert(np.array(self.DataValid)))],1)
            self.XTest=np.concatenate([np.abs(scipy.signal.hilbert(np.array(self.DataTest))),np.angle(scipy.signal.hilbert(np.array(self.DataTest)))],1)
            self.gauss="gauss"            
        elif Kind == "H_real_imag":
            self.XTrain=np.concatenate([np.real(scipy.signal.hilbert(np.array(self.DataTrain))),np.imag(scipy.signal.hilbert(np.array(self.DataTrain)))],1)
            self.XValid=np.concatenate([np.real(scipy.signal.hilbert(np.array(self.DataValid))),np.imag(scipy.signal.hilbert(np.array(self.DataValid)))],1)
            self.XTest=np.concatenate([np.real(scipy.signal.hilbert(np.array(self.DataTest))),np.imag(scipy.signal.hilbert(np.array(self.DataTest)))],1)
            self.gauss="gauss" 
        elif Kind == "LFFT":
            self.XTrain=np.log(np.abs(np.fft.fft(np.array(self.DataTrain)))[:,:60])
            self.XValid=np.log(np.abs(np.fft.fft(np.array(self.DataValid)))[:,:60])
            self.XTest=np.log(np.abs(np.fft.fft(np.array(self.DataTest)))[:,:60])
            self.gauss="gauss"
        else:
            raise ValueError("input kind for Get_X")
        
    def __getitem__(self, BS, Is_Augment=False):
        X = self.Get_Batch(BS, Is_Augment)
        if self.Kind == "Hilbert":
            X = torch.tensor(X, dtype = torch.complex64).to(DEVICE)
        else:
            X = torch.tensor(X).double().to(DEVICE)
        return X
    #def __getRes___(self)
    def Get_Batch(self, BS, Is_Augment):
        batch_select=np.random.randint(0,len(self.XTrain),BS)
        return self.XTrain[batch_select,:]
    def gt_(self, X):
        if self.Kind == "Hilbert":
            X = torch.tensor(X, dtype = torch.complex64).to(DEVICE)
        else:
            X = torch.tensor(X).double().to(DEVICE)
        return X 
    def gn_(self,X):
        return X.detach().to("cpu").numpy().squeeze()

    def Test(self,Train,Valid,Test):
        """ 
        DESCRIPTION
        
        Test model for different hyperparameters gamma
        
        
        """
        v=np.zeros((101,2,6))
        Tr=np.sort(-np.linalg.norm(Valid,axis=1))
        errorT=-np.linalg.norm(Test,axis=1)
        R_1=TestBatterie(errorT<Tr[3],self.LabelTest,errorT)
        R_2=TestBatterie(errorT<Tr[6],self.LabelTest,errorT)
        v[0,0,:]=np.concatenate(([0],R_1))
        v[0,1,:]=np.concatenate(([0],R_2))
        c=1
        for lam in self.Lambda:
            SVDD=OCC_HI(lam)
            try: 
                SVDD.Train(Train.T)
                a0=SVDD.Test(Train.T)
                a0=-1/2*SVDD.svdd.testT2.squeeze()
                a1=SVDD.Test(Valid.T)
                a1=-1/2*SVDD.svdd.testT2.squeeze()
                a2=SVDD.Test(Test.T)
                a2=-1/2*SVDD.svdd.testT2.squeeze()
                k0=np.mean(a0)-np.mean(a1)
                C0=GetOPT(SVDD)
                C2=C0-k0
                r_1=np.sum(a1<C0)/len(a1)
                r_2=np.sum(a1<C2)/len(a1)
                R_1=TestBatterie(a2<C0,self.LabelTest,a2)
                R_2=TestBatterie(a2<C2,self.LabelTest,a2)
                v[c,0,:]=np.concatenate(([r_1],R_1))  
                v[c,1,:]=np.concatenate(([r_2],R_2)) 
            except ValueError:
                v[c,0,:]=np.concatenate(([0],[0,0,0,0,0]))
                v[c,1,:]=np.concatenate(([0],[0,0,0,0,0]))
            c+=1
        return v


def GetOPT(SVDD):
    return SVDD.svdd.model["term_3"].squeeze()

class OCC_HI():
    def __init__(self, gamma=1/12, type_k="gauss",C=1):
        svdd_dic = {"positive penalty": C,
                      "negative penalty": 0.2,
                      "kernel": {"type": type_k, "width": gamma},
                      "option": {"display": 'off'}}
        self.svdd = SVDD(svdd_dic)
        self.N_min=0
        self.N_max=1
        
    def Train(self,trainDataset):
        trainDataset=trainDataset.T
        trainLabel = np.ones((len(trainDataset),1))       
        self.svdd.train(trainDataset,trainLabel)
        
            
    def Test(self,testDataset):
        testDataset=testDataset.T
        testLabel = -np.ones((len(testDataset),1))
        distance, _ = self.svdd.test(testDataset, testLabel)       
        return (distance-self.N_min)/(self.N_max-self.N_min)            
            

        
        
class SVDD():
    
    def __init__(self, parameters):
        
        """ 
        DESCRIPTION
        
        --------------------------------------------------        
        INPUT
          parameters   

             "positive penalty": positive penalty factor
             "negative penalty": negative penalty factor
             "kernel"          : kernel function         
             "option"          : some options 
             
        
        """                
        self.parameters = parameters



    def train(self, data, label):
        
        """ 
        DESCRIPTION
        
        Train SVDD model
        
        -------------------------------------------------- 
        Reference
        Tax, David MJ, and Robert PW Duin.
        "Support vector data description." 
        Machine learning 54.1 (2004): 45-66.
        
        -------------------------------------------------- 
        model = train(data, label)
        
        --------------------------------------------------        
        INPUT
        data        Training data (n*d) 
                        n: number of samples
                        d: number of features
        label       Training label (n*1)
                        positive: 1
                        negative: -1
                        
        OUTPUT
        model       SVDD hypersphere
        --------------------------------------------------
        
        """
        start_time = time.time()
        
        label = np.array(label, dtype = 'int')      
        if np.abs(np.sum(label)) == data.shape[0]:
            self.labeltype = 'single'
        else:
            self.labeltype = 'hybrid'
        
        # index of positive and negative samples
        pIndex = label[:,0] == 1
        nIndex = label[:,0] == -1
        
        # threshold for support vectors
        threshold = 1e-7
        
        # compute the kernel matrix
        K = self.getMatrix(data, data)

        # solve the Lagrange dual problem
        alf, obj, iteration = self.quadprog(K, label)
        
        # find the index of support vectors
        sv_index = np.where(alf > threshold)[0][:]

        # support vectors and alf
        sv_value = data[sv_index, :]
        sv_alf = alf[sv_index]
        
        # compute the center of initial feature space
        center = np.dot(alf.T, data)
        
        ''''
        compute the radius: eq(15)
        
        The distance from any support vector to the center of 
        the sphere is the hypersphere radius. 
        Here take the 1st support vector as an example.
        
        '''
        # the 1st term in eq(15)
        used = 0
        term_1 = K[sv_index[used], sv_index[used]]
        
        # the 2nd term in eq(15)
        term_2 = -2*np.dot(K[sv_index[used], :], alf)
        
        # the 3rd term in eq(15)
        term_3 = np.dot(np.dot(alf.T, K), alf)

        # R
        radius = np.sqrt(term_1+term_2+term_3)
        
        end_time = time.time()
        timecost = end_time - start_time
        
        # numbers of positive and negative samples
        pData = np.sum(pIndex)/data.shape[0]
        nData = np.sum(nIndex)/data.shape[0]
        
        # number of support vectors
        nSVs = sv_index.shape[0]
        
        # radio of  support vectors
        rSVs = nSVs/data.shape[0]
        
        # store the results
        self.model = {"data"      : data        ,
                      "sv_alf"    : sv_alf      ,
                      "radius"    : radius      ,
                      "sv_value"  : sv_value    ,
                      "sv_index"  : sv_index    ,
                      "nSVs"      : nSVs        ,
                      "center"    : center      ,
                      "term_3"    : term_3      ,
                      "alf"       : alf         ,  
                      "K"         : K           ,
                      "pIndex"    : pIndex      ,
                      "nIndex"    : nIndex      ,
                      "obj"       : obj         ,
                      "iteration" : iteration   ,
                      "timecost"  : timecost    ,
                      "pData"     : pData       ,
                      "nData"     : nData       ,
                      "rSVs"      : rSVs        ,
                      }
        
        # compute the training accuracy
        display_ = self.parameters["option"]["display"]
        self.parameters["option"]["display"] = 'off'
        _, accuracy = self.test(data, label)
        self.parameters["option"]["display"] = display_      
        self.model["accuracy"] = accuracy
        
        # display training results       
        if self.parameters["option"]["display"] == 'on':
            print('\n')
            print('*** SVDD model training finished ***\n')
            print('iter             = %d'       % self.model["iteration"])
            print('time cost        = %.4f s'   % self.model["timecost"])
            print('obj              = %.4f'     % self.model["obj"])
            print('pData            = %.4f %%'  % (100*self.model["pData"]))
            print('nData            = %.4f %%'  % (100*self.model["nData"]))
            print('nSVs             = %d'       % self.model["nSVs"])
            print('radio of nSVs    = %.4f %%'  % (100*self.model["rSVs"]))
            print('accuracy         = %.4f %%'  % (100*self.model["accuracy"]))
            print('\n')
  
    def test(self, data, label):
    
        """ 
        DESCRIPTION
        
        Test the testing data using the SVDD model
    
        distance = test(model, Y)
        
        --------------------------------------------------        
        INPUT
        data        Test data (n*d) 
                        n: number of samples
                        d: number of features
        label       Test label (n*1)
                        positive: 1
                        negative: -1
            
        OUTPUT
        distance    Distance between the test data and hypersphere
        --------------------------------------------------
        
        """    
        
        start_time = time.time()
        n = data.shape[0]
        
        # compute the kernel matrix
        K = self.getMatrix(data, self.model["data"])
        
        # the 1st term
        term_1 = self.getMatrix(data, data)
        
        # the 2nd term
        tmp_1 = -2*np.dot(K, self.model["alf"])
        term_2 = np.tile(tmp_1, (1, n))
        self.testT2=tmp_1

        # the 3rd term
        term_3 =  self.model["term_3"]
        
        # distance
        distance = np.sqrt(np.diagonal(term_1+term_2+term_3))
        
        # predicted label
        predictedlabel = np.mat(np.ones(n)).T
        fault_index = np.where(distance > self.model["radius"])[1][:]
        predictedlabel[fault_index] = -1
            
        # compute prediction accuracy
        accuracy = np.sum(predictedlabel == label)/n
        
        end_time = time.time()
        timecost = end_time - start_time
        if self.parameters["option"]["display"] == 'on':
        # display test results
            print('\n')
            print('*** SVDD model test finished ***\n')
            print('time cost        = %.4f s'   % timecost)
            print('accuracy         = %.4f %%'  % (100*accuracy))
            print('\n')
        
        
        return distance, accuracy 

    def quadprog(self, K, label):
    
        """ 
        DESCRIPTION
        
        Solve the Lagrange dual problem
        
        quadprog(self, K, label)
        
        --------------------------------------------------
        INPUT
        K         Kernel matrix
        label     training label
        
                        
        OUTPUT
        alf       Lagrange multipliers
        
        --------------------------------------------------
        
        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b                    
        --------------------------------------------------
        
        """ 
        solvers.options['show_progress'] = False
        
        label = np.mat(label)
        K = np.multiply(label*label.T, K)
        
        # P
        n = K.shape[0]
        P = K+K.T
        
        # q
        q = -np.multiply(label, np.mat(np.diagonal(K)).T)

        # G
        G1 = -np.eye(n)
        G2 = np.eye(n)
        G = np.append(G1, G2, axis=0)
        
        # h
        h1 = np.mat(np.zeros(n)).T # lb
        h2 = np.mat(np.ones(n)).T
        if self.labeltype == 'single':
            h2[label == 1] = self.parameters["positive penalty"]
        
        if self.labeltype == 'hybrid':
            h2[label == 1] = self.parameters["positive penalty"]
            h2[label == -1] = self.parameters["negative penalty"]

            
        h = np.append(h1, h2, axis=0)
        
        # A, b
        A = np.mat(np.ones(n))
        b = 1.
        
        #
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        
        #
        sol =solvers.qp(P, q, G, h, A, b)
        alf = np.array(sol['x'])
        obj = np.array(sol['dual objective'])
        iteration = np.array(sol['iterations'])

        return alf, obj, iteration

    def getMatrix(self, X, Y):
    
        """ 
        DESCRIPTION
        
        Compute kernel matrix 
        
        K = getMatrix(X, Y)
        
        -------------------------------------------------- 
        INPUT
        X         data (n*d)
        Y         data (m*d)

        OUTPUT
        K         kernel matrix 
        -------------------------------------------------- 
                        
                            
        type   -  
        
        linear :  k(x,y) = x'*y+c
        poly   :  k(x,y) = (x'*y+c)^d
        gauss  :  k(x,y) = exp(-s*||x-y||^2)
        tanh   :  k(x,y) = tanh(g*x'*y+c)
        lapl   :  k(x,y) = exp(-s*||x-y||)
           
        degree -  d
        offset -  c
        width  -  s
        gamma  -  g
        
        --------------------------------------------------      
        ker    - 
        
        ker = {"type": 'gauss', "width": s}
        ker = {"type": 'linear', "offset": c}
        ker = {"type": 'ploy', "degree": d, "offset": c}
        ker = {"type": 'tanh', "gamma": g, "offset": c}
        ker = {"type": 'lapl', "width": s}
    
        """
        def gaussFunc():
            
            if self.parameters["kernel"].__contains__("width"):
                s =  self.parameters["kernel"]["width"]
            else:
                s = 2
            K = smp.rbf_kernel(X, Y, gamma=s)

                
            return K
            
        def linearFunc():
            
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 0

            K = smp.linear_kernel(X, Y)+c
            
            return K
        
        def ployFunc():
            if self.parameters["kernel"].__contains__("degree"):
                d =  self.parameters["kernel"]["degree"]
            else:
                d = 2
                
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 0
                
            K = smp.polynomial_kernel(X, Y, degree=d, gamma=None, coef0=c)
            
            return K
             
        def laplFunc():
            
            if self.parameters["kernel"].__contains__("width"):
                s =  self.parameters["kernel"]["width"]
            else:
                s = 2
            K = smp.laplacian_kernel(X, Y, gamma=s)

            return K
    
        def tanhFunc():
            if self.parameters["kernel"].__contains__("gamma"):
                g =  self.parameters["kernel"]["gamma"]
            else:
                g = 0.01
                
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 1
            
            K = smp.sigmoid_kernel(X, Y, gamma=g, coef0=c)

            return K
        
        def cgaussFunc():
            
            if self.parameters["kernel"].__contains__("width"):
                s =  self.parameters["kernel"]["width"]
            else:
                s = 2
                
            Kr = smp.rbf_kernel(X.real, Y.real, gamma=s) + smp.rbf_kernel(X.imag, Y.imag, gamma=s)
            Ki = smp.rbf_kernel(X.real, Y.imag, gamma=s) - smp.rbf_kernel(X.imag, Y.real, gamma=s)

                
            return np.abs(Kr + 1j * Ki)
             
        
        
        kernelType = self.parameters["kernel"]["type"]
        switcher = {    
                        "gauss"   : gaussFunc  ,        
                        "linear"  : linearFunc ,
                        "ploy"    : ployFunc   ,
                        "lapl"    : laplFunc   ,
                        "tanh"    : tanhFunc   ,
                        "cgauss"  : cgaussFunc ,
  
                     }
        
        return switcher[kernelType]()


    def Save_Model(self, Path):
        with open(Path, 'wb') as fp:
            pickle.dump(self.model, fp, protocol=pickle.HIGHEST_PROTOCOL)

        
    def Load_Model(self, Path):
        with open(Path, 'rb') as fp:
            self.model = pickle.load(fp)
            
            
    def Min_Max_update(self,v_min,v_max):
        self.model["v_min"]=v_min
        self.model["v_max"]=v_max
        
    def Min_Max_get(self):
        return self.model["v_min"], self.model["v_max"]
    
    
class AEnet(nn.Module):
    def __init__(self,In):
        super(AEnet, self).__init__()
        #self.deep=len(In)-1
        self.autoencoder=nn.Sequential(nn.Linear(In,64),
                                  nn.ReLU(),
                                  nn.Linear(64,64),
                                  nn.ReLU(),
                                  nn.Linear(64,32),
                                  nn.ReLU(),
                                  nn.Linear(32,64),
                                  nn.ReLU(),
                                  nn.Linear(64,64),
                                  nn.ReLU(),
                                  nn.Linear(64,In))
    def forward(self,x):   
        return self.autoencoder(x)


class Comp_ReLU(nn.Module):
    def __init__(self,typeA=1):
        super(Comp_ReLU, self).__init__()
        self.typeA=typeA
        if typeA=="modReLU"or typeA=="EAD":
            self.biais = nn.Parameter(data=0.1*torch.ones(1,dtype=torch.complex64).clone() ,requires_grad=True)   
    def forward(self,z):
        if self.typeA=="CReLU":#1:#CReLU
             return F.relu(z.real)+ 1j*F.relu(z.imag)
        elif self.typeA=="modReLU":#2:#modReLU
            return F.relu(torch.abs(z)-torch.abs(self.biais))*torch.exp(1j*torch.angle(z))
        elif self.typeA=="EAD":#3:#EAD
            return (1-torch.exp(-torch.abs(self.biais)*torch.abs(z)**2))*torch.exp(1j*torch.angle(z))   
class AEnet_c(nn.Module):
    def __init__(self,In,fun):
        super(AEnet_c, self).__init__()
        self.autoencoder=nn.Sequential(nn.Linear(In,64, dtype=torch.complex64),
                                  Comp_ReLU(fun),
                                  nn.Linear(64,64, dtype=torch.complex64),
                                  Comp_ReLU(fun),
                                  nn.Linear(64,32, dtype=torch.complex64),
                                  Comp_ReLU(fun),
                                  nn.Linear(32,64, dtype=torch.complex64),
                                  Comp_ReLU(fun),
                                  nn.Linear(64,64, dtype=torch.complex64),
                                  Comp_ReLU(fun),
                                  nn.Linear(64,In, dtype=torch.complex64))
    def forward(self,x):      
        return self.autoencoder(x)

def GetF1Score(M):
    F1_id = 3
    r1=M[0][1][F1_id]
    M=M[1:]
    r2=[0,0]
    for style in range(2):
        c=np.array(M)[:,style,0]
        c[c==0]=1
        c=np.abs(c-0.05)
        cm=np.min(c)
        idx=np.max(np.where(c==cm))
        r2[style]=np.array(M)[idx,style,F1_id]
    return r1, r2[0], r2[1]