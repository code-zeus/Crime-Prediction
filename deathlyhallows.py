#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:43:08 2018

@author: kartikey
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

df=pd.read_csv("/home/kartikey/Desktop/Project ML/test.csv")
df1=df["Date"]
df2=df["IUCR"]
df3=df["Location Description"]
serial_date=[0]*100
serial_time=[0]*100
serial_iucr=[str(0)]*1
serial_loc=[str(0)]*1
temp=[str(0)]*100
temp1=[str(0)]*100
arr=[31,29,31,30,31,30,31,31,30,31,30,31]
for i in range(1,12):
    arr[i]=arr[i]+arr[i-1]
for i in range(1,100):
    a=df1[i]
    p=int(a[0])*10+int(a[1])
    p=p-1
    serial_date[i]=serial_date[i]+arr[p-1]+int(a[3])*10+int(a[4])
    serial_time[i]=serial_time[i]+(int(a[11])*10+int(a[12]))*3600+(int(a[14])*10+int(a[15]))*60+(int(a[17])*10+int(a[18]))
    if(a[20]=="P"):
        serial_time[i]=serial_time[i]+(12*3600)
for i in range(1,100):
    temp[i]=df2[i]
    temp1[i]=df2[i]
temp.sort()
for i in range(1,100):
    if(temp[i]!=temp[i-1]):
        serial_iucr.append(temp[i])
temp2=[0]*100
for i in range(1,100):
    temp[i]=df3[i]
    temp2[i]=df3[i]
temp.sort()
for i in range(1,100):
    if(temp[i]!=temp[i-1]):
        serial_loc.append(temp[i])
mapped_loc=[0]*100
mapped_iucr=[0]*100
for i in range(1,100):
   mapped_iucr[i]= serial_iucr.index(temp1[i])
   mapped_loc[i]=serial_loc.index(temp2[i])
normalized_date=[0]*len(serial_date)
normalized_time=[0]*len(serial_time)
normalized_iucr=[0]*len(mapped_iucr)
normalized_loc=[0]*len(mapped_loc)
for i in range(1,len(serial_date)):
    normalized_date[i]=i/len(serial_date)
for i in range(1,len(serial_time)):
    normalized_time[i]=i/len(serial_time)
for i in range(1,len(mapped_iucr)):
    normalized_iucr[i]=mapped_iucr[i]/len(serial_iucr)
for i in range(1,len(mapped_loc)):
    normalized_loc[i]=mapped_loc[i]/len(serial_loc)
Xdf = pd.DataFrame(np.column_stack([normalized_loc,normalized_date,normalized_time,normalized_iucr]),columns=['Location', 'Date', 'Time','IUCR'])
train, test = train_test_split(Xdf, test_size=0.2)
n=len(serial_iucr)
y = np.array(train['IUCR'])
X = np.array(train.iloc[:,0:3])
y_test = np.array(test['IUCR'])
X_test = np.array(test.iloc[:,0:3])
kmeans = KMeans(n_clusters=n) 
kmeans.fit(X,y)
correct = 0
label=kmeans.labels_
index=train.index.values
iucr=[0]*80
for i in range(len(X)):
    iucr[i]=serial_iucr[mapped_iucr[index[i]]]
new_df = pd.DataFrame(np.column_stack([label,iucr]),columns=['Label', 'IUCR'])
for i in range(len(X_test)):
    predict_me = np.array(X_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    print(prediction)
    if iucr[i]==:
        correct += 1
print(correct/len(X))
print(kmeans.labels_)












