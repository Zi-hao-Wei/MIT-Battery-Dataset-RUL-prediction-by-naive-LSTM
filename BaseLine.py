from GetData import GetData
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.signal
from sklearn import preprocessing
import tensorflow as tf 
from keras.models import load_model
from keras.layers import Input,LSTM,Dense 
from keras import Sequential
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def GetTestFeatures():
    Data=GetData()
    TrainSet=Data[12]

    # print(TrainSet['cycle_life'])

    features=[]
    l=[]
    for i in TrainSet['summary']:
        if(i == "QD"):
            # print(i)
            # l.append(i)
            d=TrainSet['summary'][i]
            d=d.reshape(-1,1)
            min_max_scaler = preprocessing.MinMaxScaler()    
            x_minmax = min_max_scaler.fit_transform(d)
            # x_minmax=d
            features.append(x_minmax.squeeze())

    features=np.array(features)
    print(features.shape)
    features=features.T
    print(features.shape)
    # features=features.reshape((849,-1))
    return features

def sliceWindow(data,step): 
    X,y=[],[]
    for i in range(0,849-step,1): 
        end=i+step 
        oneX,oney=data[i:end,:],data[end,:] 
        X.append(oneX) 
        y.append(oney) 
    return np.array(X),np.array(y) 

def dataSplit(dataset,step,ratio=0.90): 
    datasetX,datasetY=sliceWindow(dataset,step)
    train_size=int(len(datasetX)*ratio)
    print(train_size)

    X_train,y_train=datasetX[0:train_size,:],datasetY[0:train_size,:]

    X_test,y_test=datasetX[train_size:len(datasetX),:],datasetY[train_size:len(datasetX),:]

    X_train=X_train.reshape(X_train.shape[0],step,-1)
    X_test=X_test.reshape(X_test.shape[0],step,-1)
    print('X_train.shape: ',X_train.shape)
    print('X_test.shape: ',X_test.shape)
    print('y_train.shape: ',y_train.shape)
    print('y_test.shape: ',y_test.shape)
    return X_train,X_test,y_train,y_test

def seq2seqModel(X,step):
    model=Sequential() 
    model.add(LSTM(256, activation='relu', return_sequences=True,input_shape=(step,X.shape[2]))) 
    model.add(LSTM(256, activation='relu'))
    model.add(Dense(X.shape[2])) 
    model.compile(optimizer='adam', loss='mse') 
    return model 

def predictFuture(model,dataset,features,step,next_num): 
    lastOne=(dataset[len(dataset)-step:len(dataset)]).reshape(-1,features)
    backData=lastOne.tolist()
    next_predicted_list=[]
    for i in range(next_num):
        one_next=backData[len(backData)-step:]
        one_next=np.array(one_next).reshape(1,step,features)
        next_predicted=model.predict([one_next])
        next_predicted_list.append(next_predicted[0].tolist())
        backData.append(next_predicted[0])
    return next_predicted_list 



if __name__=="__main__":
    step=20
    # features=8
    features=GetTestFeatures()
    X_train,X_test,y_train,y_test=dataSplit(features,step)
    model=seq2seqModel(X_train,step) 
    model.fit(X_train,y_train,epochs=100,verbose=1) 
    model.save('m1.h5')
    # y_pre=model.predict(X_test,verbose=1)
    # print('y_pre: ',y_pre) 
    # future_list=predictFuture(model,features,1,step,250)
    y1=[]
    y2=[]
    t1=X_train[-1]
    print(t1.shape)
    for i in range(83):
        now=t1.reshape(1,step,1)
        temp=model.predict(now)
        t1[0:step-1]=t1[1:step]
        t1[step-1]=temp
        y1.append(temp[0,0])
        y2.append(y_test[i,0])

    plt.plot(y1,label="predicted")
    plt.plot(y2,label="True")
    print(y1)
    print(y2)
    plt.legend()
    plt.show()
