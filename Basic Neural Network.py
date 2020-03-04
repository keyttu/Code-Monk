import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
def sigmoid(k):
    h = math.exp(-k)+1
    return 1/h
X = np.zeros(100,dtype='int64')
y = np.zeros(50,dtype='int64')
for i in range(len(X)):
    X[i] = random.randint(-4,4)
X = X.reshape(50,2)
y = y.reshape(50,1)
for i in range(len(X)):
    if(X[i][0]>=0 and X[i][1]>=0):
        y[i]=0
    if(X[i][0]<=0 and X[i][1]>=0):
        y[i]=1
    if(X[i][0]<=0 and X[i][1]<=0):
        y[i]=2
    if(X[i][0]>=0 and X[i][1]<=0):
        y[i]=3
y_output = np.zeros(200).reshape(50,4)
for i in range(len(y_output)):
    h=y[i]
    y_output[i][h]=1
''' NOW THE DATASET IS READY
and coming to Structure of Neural net we have 2 input
neurons and 5 hidden layer neurons and 4 output neurons'''
W = np.zeros(10).reshape(2,5)
h = np.zeros(5).reshape(5,1)
hb= np.zeros(5).reshape(5,1)
Z = np.zeros(20).reshape(5,4)
O = np.zeros(4).reshape(4,1)
Ob = np.zeros(4).reshape(4,1)
loss = np.zeros(4).reshape(4,1)
n = 0.1
''' First we need to initialize the weights by random'''
W = W.ravel()
for i in range(10):
    W[i] = random.randint(-3,3)
W = W.reshape(2,5)
Z = Z.ravel()
for i in range(20):
    Z[i] = random.randint(-3,3)
Z = Z.reshape(5,4)
for i in range(len(hb)):
    hb[i] = random.randint(-5,5)
for i in range(len(Ob)):
    Ob[i] = random.randint(-5,5)
for g in range(len(X)):   
    '''FeedForward path 1'''
    A = X[g].reshape(2,1) #CHANGE
    h_temp = (W.T).dot(A)
    h_temp = h_temp + hb
    for i in range(len(h)):
        h[i]= sigmoid(h_temp[i])
    O_temp = (Z.T).dot(h)
    O_temp = O_temp+Ob
    for i in range(len(O)):
        O[i] = sigmoid(O_temp[i])
   
    ''' Backpropagation 1'''
    keyttu=0
    while(keyttu<200):
        for i in range(len(O)):
            loss[i] = O[i]-y_output[g][i]
        for i in range(len(Z)):
            for j in range(len(Z.T)):
                Z[i][j] = Z[i][j] - n*2*loss[j]*O[j]*O[j]*(math.exp(-O_temp[j]))*h[i]
       
        for i in range(len(Z.T)):
                Ob[i] = Ob[i] - n*2*loss[i]*O[i]*O[i]*(math.exp(-O_temp[i]))
       
        ''' Backpropagation 2'''
        for i in range(len(W)):
            for j in range(len(W.T)):
                a=0
                for k in range(len(Z.T)):
                    a = a+2*loss[k]*O[k]*O[k]*(math.exp(-O_temp[k]))*Z[j][k]*h[j]*h[j]*(math.exp(-h_temp[j]))
                W[i][j] = W[i][j] - n*a*X[g][i] #CHANGE
               
        for j in range(len(Z)):
            a=0
            for k in range(len(Z.T)):
                a = a+2*loss[k]*O[k]*O[k]*(math.exp(-O_temp[k]))*Z[j][k]*h[j]*h[j]*(math.exp(-h_temp[j]))
            hb[j]= hb[j]-n*a
       
        '''FeedForward path 1'''
        A = X[g].reshape(2,1) #CHANGE
        h_temp = (W.T).dot(A)
        h_temp = h_temp + hb
        for i in range(len(h)):
            h[i]= sigmoid(h_temp[i])
        O_temp = (Z.T).dot(h)
        O_temp = O_temp+Ob
        for i in range(len(O)):
            O[i] = sigmoid(O_temp[i])
        keyttu=keyttu+1

test = np.array([[-100],[-100]])
h_temp = (W.T).dot(test)
h_temp = h_temp + hb
for i in range(len(h)):
    h[i]= sigmoid(h_temp[i])
O_temp = (Z.T).dot(h)
O_temp = O_temp+Ob
for i in range(len(O)):
    O[i] = sigmoid(O_temp[i])
print('The quadrant is ',O.argmax()+1)
