from GetData import GetData,GetSmoothedData
import numpy as np 
import matplotlib.pyplot as plt 
# import pandas as pd 
import scipy.signal
from sklearn import preprocessing
import tensorflow as tf 
from keras.models import load_model
from keras.layers import Input,LSTM,Dense 
from keras import Sequential
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def GetFeatures(idx,f=["QD"]):
    Data=GetSmoothedData()
    TrainSet=Data[idx]
    features=[]
    for i in TrainSet['summary']:
        if(i in f):
            print(i)
            d=TrainSet['summary'][i]
            d=d.reshape(-1,1)
            min_max_scaler=preprocessing.MinMaxScaler()    
            x_minmax=min_max_scaler.fit_transform(d)
            features.append(x_minmax.squeeze())
    features=np.array(features)
    features=features.T    
    return features

def BuildSet(Dataset,step=50):
    Cyclelife=len(Dataset)
    print(Cyclelife)
    x=[]
    y=[]
    for i in range(0,len(Dataset)-step+1):
        oneX=Dataset[i:i+step]
        oneY=(Cyclelife-step-i)/Cyclelife
        x.append(oneX)
        y.append(oneY)
    return x,y



def seq2seqModel(X,step):
    model=Sequential() 
    model.add(LSTM(256, activation='relu', return_sequences=True,input_shape=(step,X.shape[2]))) 
    model.add(LSTM(256, activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss="mse") 
    return model 


def test(f,idx=14,filename="m1.h5"):
    model=load_model(filename)
    model.summary()
    features=GetFeatures(idx,f)
    testX,testY=BuildSet(features,step)
    testX=np.array(testX)
    testY=np.array(testY)
    testX=testX.reshape(testX.shape[0],step,-1)
    
    ans=model.predict(testX)

    
    plt.figure(1)
    Data=GetData()
    TrainSet=Data[idx]
    capacity=[]
    IR=[]
    for i in TrainSet['summary']:
        if(i in ["QD"]):
            d=TrainSet['summary'][i]
            capacity=d
        if(i in ["IR"]):
            d=TrainSet['summary'][i]
            IR=d

    # plt.subplot(311)
    # plt.plot(capacity,label="QD")
    # plt.xlabel("Cycle")
    # plt.ylabel("Discharge Capacity (Ah)")
    # plt.legend()

    # plt.subplot(312)
    # plt.plot(IR,label="IR")
    # plt.xlabel("Cycle")
    # plt.ylabel("Internal Resistance (Ohm)")
    # plt.legend()

    # plt.subplot(313)
    testY*=len(capacity)
    delta=[]
    for i in range(len(ans)):
        ans[i]=i/(1-ans[i])*ans[i]
        delta.append(ans[i]-testY[i])
    x=np.arange(step,len(capacity)+1)
    # plt.plot(x,ans,label="Remain Cycles Predict")
    plt.plot(x,delta)

    plt.xlabel("Last cycle")
    plt.ylabel("Cycle Error")
    plt.ylim((-100,100))
    plt.legend()
    plt.show()

def train(step,f,filename):
    trainX,trainY=[],[]
    for i in [1,8,12,18,22]:
        features=GetFeatures(i,f)
        trainXTemp,trainYTemp=BuildSet(features,step)
        trainX=trainX+trainXTemp
        trainY=trainY+trainYTemp
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    print(trainX.shape)

    trainX=trainX.reshape(trainX.shape[0],step,-1)

    model=seq2seqModel(trainX,step)

    model.fit(trainX, trainY, epochs=50)
    model.save(filename)

if __name__=="__main__":
    step=100
    f=["QD","IR"]
    filename="m4.h5"
    idx=14
    # train(step,f,filename)
    test(f,idx,filename)
