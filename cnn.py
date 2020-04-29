# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 04:37:43 2020

@author: Apel
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
#Load data using keras dataset
#from keras.datasets import mnist
#
#
#(x_train,y_train),(x_test,y_test)=mnist.load_data()
#
#print(x_train.head())
##load data using local file 
test=pd.read_csv("mnist_test.csv")
train=pd.read_csv("mnist_train.csv")
##preprocess 
print(test.head())
print("tarin head data")
print(train.head())
Y_train=train["label"]
X_train=train.iloc[:,1:]
Y_test=test["label"]
X_test=test.iloc[:,1:]
## normalization
X_train=X_train/255
X_test=X_test/255
X_train=X_train.values.reshape(-1,28,28,1)
X_test=X_test.values.reshape(-1,28,28,1)
vis=X_train.reshape(60000,28,28)
plt.imshow(vis[5,:,:])
plt.title(Y_train[5])
## Encoding
from keras.utils import to_categorical
Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)
##CNN
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from keras.models import Sequential
model=Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters=4,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
hist=model.fit(X_train,Y_train,batch_size=4000,epochs=25,validation_data=(X_test,Y_test))
##Virtualization
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.plot(hist.history["loss"],label=" Loss")
plt.show()
plt.plot(hist.history["val_accuracy"],label="Validation Accuracy")
plt.plot(hist.history["accuracy"],label=" Accuracy")

## save hist 
hist_df=pd.DataFrame(hist.history)
save_path="cnn_mnist_hist.csv"
with open(save_path, 'w') as f:
    hist_df.to_csv(f)        
# load hist
h=pd.read_csv("cnn_mnist_hist.csv")
h=h.iloc[:,1:]

## Virtualization load data 
plt.plot(h.iloc[:,0],label = "Vall Loss")
plt.plot(h.iloc[:,2],label = " Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h.iloc[:,1],label = "Vall Accuracyy")
plt.plot(h.iloc[:,3],label = "Accuracy")

plt.legend()
plt.show()