import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal

def GetData():
    batch3 = pickle.load(open(r'.\Data\batch3V.pkl','rb'))
    # remove noisy channels from batch3
    del batch3['b3c37']
    del batch3['b3c2']
    del batch3['b3c23']
    del batch3['b3c32']
    del batch3['b3c38']
    del batch3['b3c39']
    # numBat = len(batch3.keys())
    bat_dict = {**batch3}
    Data=[]
    for i in bat_dict.keys():
        Data.append((bat_dict[i],bat_dict[i]["cycle_life"]))

    Data=sorted(Data,key=lambda x:x[1])
    Data=Data[0:24]
    Processed=[]
    for i,_ in Data:
        Processed.append(i)
    return Processed

def GetSmoothedData(width=51,degree=6):
    Data=GetData()
    for i in Data:
        for j in i['summary']:
            if j != "cycle":
                original=i['summary'][j]
                smoothed=scipy.signal.savgol_filter(original,width,degree)
                i['summary'][j]=smoothed
    return Data

def GetProcessedData():
    Data=GetSmoothedData()

if __name__=="__main__":
    Data=GetSmoothedData()
    for i in Data:
        original=i['summary']['QD']
        plt.plot(i['summary']['cycle'], original)

    plt.xlabel('cycle')
    plt.ylabel('capacity')
    plt.show()