# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:54:28 2019

@author: Pengfei Fan

SCALA: Semi-supervised Confidence-based Adaptively Learnable Approach toward Scalable Data Transmission through Multimode Fibers

"""

import os
import numpy as np
import time
import keras
from keras.models import Model
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from sklearn.metrics import accuracy_score, f1_score
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

epochs = 200
basepath = '../../'

# load x and y
x = np.load(basepath+'x.p.npy')
y = np.load(basepath+'y.p.npy')

imagesize = x.shape[1]
outsize = y.shape[1]
input_shape=(imagesize, imagesize,1)

y = y.reshape(y.shape[0],y.shape[1]*y.shape[2])

dataValues = x.copy()
dataLabels = y.copy()
print('data set size:')
print(dataValues.shape)
print(dataLabels.shape)

reset = False #change

condition = {0: 'no', 1: 'fixed', 2: 'dynamic'}# 0:No, 1:fixed, 2:dynamic
confidence = 2 #change: 0 = Sliding SSL, so only 1 or 2
denoising = 2 #change: 0, 1 or 2

sizeOfBatch = 500  #change
initialPercentage = 2000/x.shape[0]  #change
initialLabeledData = int(x.shape[0]*initialPercentage)
batches = int((x.shape[0]*(1-initialPercentage)/sizeOfBatch))
sizeOfWindow = initialLabeledData

print('sizeOfWindow:', sizeOfWindow)
print('sizeOfBatch:', sizeOfBatch)
print('batches:', batches)
print('reset:', reset)
print('confidence:', condition[confidence])
if confidence == 1:
    excludingPercentage = 0.1   #change
else:
    excludingPercentage = 'unknown'
print('cuttingPercentage:', excludingPercentage)

print('denoising:', condition[denoising])
if denoising != 0:
    denoisingProbability = 0.65  #change
else:
    denoisingProbability = 'no'
print('denoisingProbability:', denoisingProbability)

if reset == True:
    path = './save/reset/'
    if os.path.exists(path) == False:
        os.makedirs(path)
else:
    path = './save/continue/'
    if os.path.exists(path) == False:
        os.makedirs(path)
        
savepath = path+'%s'%(initialLabeledData)+'+'+'%s'%(sizeOfWindow)+'('+'%s'%(sizeOfBatch)+')'+'x'+'%s/'%(batches)+'a=%s_'%(condition[confidence]+'_'+str(excludingPercentage))+'p=%s/'%(condition[denoising]+'_'+str(denoisingProbability))
if os.path.exists(savepath) == False:
    os.makedirs(savepath)

log_path = savepath+'log/'
if os.path.exists(log_path) == False:
    os.makedirs(log_path)
log_path = log_path+'log.csv'

evaluationl_path = savepath+'evaluation/'
if os.path.exists(evaluationl_path) == False:
    os.makedirs(evaluationl_path)

step_model_path = savepath+'step/'
if os.path.exists(step_model_path) == False:
    os.makedirs(step_model_path)

def createCNN(outsize, input_shape):

    model_input = Input(shape=input_shape)
    
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(model_input)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x)
    
    x = Flatten()(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x)
    x = Dense(outsize*outsize, activation='sigmoid')(x)
    
    cnn = Model(inputs=model_input, outputs=x)

    return cnn
    
ada=keras.optimizers.Adadelta(lr=1e-2)

def get_callbacks():
#     checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1,
#                                  save_best_only=True, save_weights_only=False, mode='auto', period=1)
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, verbose=1, mode='auto')
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
#                                   verbose=1, mode='auto', cooldown=0, min_lr=1e-7)
    csv_logger = CSVLogger(log_path, append=True)
#     return [checkpoint, csv_logger,
#             reduce_lr, early_stopping
#             ]
    return [csv_logger]

def loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength):
    X = np.copy(dataValues[initialDataLength:finalDataLength])
    y = np.copy(dataLabels[initialDataLength:finalDataLength])
    return X, y

def c_evaluate(y_actual, y_predicted):
    accuracy = []
    f1score = []
    for i in range(0,y_actual.shape[0]):
        a = accuracy_score(y_actual[i], y_predicted[i])*100
        accuracy.append(a)
        f1 = f1_score(y_actual[i], y_predicted[i], average='binary')
        f1score.append(f1)
    return accuracy, np.mean(accuracy), f1score, np.mean(f1score)

def r_evaluate(y_actual, y_predicted):
    corr = []
    mse = []
    ssim = []
    for i in range(0,y_actual.shape[0]):
        c1 = np.corrcoef(y_actual[i], y_predicted[i])[0,1]  
        p1 = y_predicted[i].reshape(outsize,outsize)
        y1 = y_actual[i].reshape(outsize,outsize).astype(p1.dtype)   
        m1 = compare_mse(y1, p1)
        if outsize < 7:
            y1 = y1.repeat(2,axis=1).repeat(2,axis=0)  
            p1 = p1.repeat(2,axis=1).repeat(2,axis=0)  
        s1 = compare_ssim(y1, p1, data_range = p1.max() - p1.min())
        corr.append(c1)
        mse.append(m1)
        ssim.append(s1)
    return corr, np.mean(corr), mse, np.mean(mse), ssim, np.mean(ssim)

'''
why 'if outsize < 7:'?
https://github.com/scikit-image/scikit-image/issues/1998

when your input image is too small (eg. one of its size is less than 7) it will also rise this error even if setting multichannel=True,
this is caused by

scikit-image/skimage/measure/_structural_similarity.py

Lines 144 to 146 in e6769c5

 if np.any((np.asarray(X.shape) - win_size) < 0): 
     raise ValueError( 
         "win_size exceeds image extent.  If the input is a multichannel " 
'''
    
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]
    
# Confidence level
def confidenceFactor(arr):
    median = np.ones(arr.shape[-1])*0.5
    W = []
    for i in range(0, arr.shape[0]):
        w = np.sum(np.abs(arr[i]-median)+0.5)/arr.shape[-1]
        W.append(w)   
    meanW = np.mean(W)
    return W, meanW

#squared hellinger distance
def hellinger(p, q):
    dist = np.sum(np.square(np.sqrt(p) - np.sqrt(q)))/2
    return dist

def cuttingPercentage(Xt_1, Xt, t=None, bias=0):
    res = []
    for i in range(Xt_1.shape[1]):
        for j in range(Xt_1.shape[2]):
            P = Xt_1[:, i , j, :]
            Q = Xt[:, i , j, :]
            first_edge = np.min([np.min(P), np.min(Q)])
            last_edge = np.max([np.max(P), np.max(Q)])
            n_equal_bins = int(np.sqrt(len(Xt_1)))
            bin_edges = np.linspace(first_edge, last_edge, num=n_equal_bins + 1, endpoint=True)
            hP = np.histogram(P, bins=bin_edges, density=True)
            hQ = np.histogram(Q, bins=bin_edges, density=True)
            h = hellinger(hP[0]*np.diff(hP[1]), hQ[0]*np.diff(hQ[1]))
            res.append(h)
    H = np.mean(res)
    alpha = H + bias
    if alpha > 0.5:
        alpha = 0.5
    elif alpha < 0.001:
        alpha = 0.001
    print(t, H, alpha)
    return H, alpha #percentage of excluding
    
arrCorr = []
meanCorr = []
arrMse = []
meanMse = []
arrSsim = []
meanSsim = []

arrAcc = []
meanAcc = []
arrF1 = []
meanF1 = []

arrCon = []
meanCon = []
arrDenPro = [denoisingProbability]

updatingTime = []
predictingTime = []
arrClf = []
initialDataLength = 0
finalDataLength = initialLabeledData
# ***** Box 1 *****
#Initial labeled data
X, y = loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength)
clf = createCNN(outsize = outsize, input_shape = input_shape)
clf.compile(loss = keras.losses.binary_crossentropy, optimizer= ada)

for t in range(batches):
    print('step:', t)
    print('initial:', initialDataLength)
    print('final:', finalDataLength)

    # sliding 
    initialDataLength = finalDataLength
    finalDataLength = finalDataLength + sizeOfBatch
    Ut, yt = loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength)
    # updatingTime
    start1 = time.time()
    
    # this training is to predict the next unlabled batch
    clf.fit(X, y, batch_size = 32, epochs = epochs, callbacks=get_callbacks(), validation_data=(Ut, yt))
    
    arrClf.append(clf)
    save_model(clf, step_model_path+'model_'+str(int(t))+'.hdf5')
    
    # predictingTime
    start2 = time.time()
    predicted = clf.predict(Ut)#continuous value
    end2 = time.time()
    predictingTime.append(end2-start2)
    np.savetxt(evaluationl_path+'predictingTime.txt', predictingTime)
    
    W = confidenceFactor(predicted)
    arrCon.append(W[0])
    meanCon.append(W[1])
    np.savetxt(evaluationl_path+'arrCon.txt', arrCon)
    np.savetxt(evaluationl_path+'meanCon.txt', meanCon)
    print('mean confidence:', W[1])

    # Evaluating regression
    evaluationR = r_evaluate(yt, predicted)
    arrCorr.append(evaluationR[0])
    meanCorr.append(evaluationR[1])
    arrMse.append(evaluationR[2])
    meanMse.append(evaluationR[3])
    arrSsim.append(evaluationR[4])
    meanSsim.append(evaluationR[5])
    np.savetxt(evaluationl_path+'arrCorr.txt', arrCorr) 
    np.savetxt(evaluationl_path+'meanCorr.txt', meanCorr)
    np.savetxt(evaluationl_path+'arrMse.txt', arrMse) 
    np.savetxt(evaluationl_path+'meanMse.txt', meanMse)
    np.savetxt(evaluationl_path+'arrSsim.txt', arrSsim) 
    np.savetxt(evaluationl_path+'meanSsim.txt', meanSsim)
    
    predsb = predicted.copy()
    predsb[predsb>=0.5] = 1
    predsb[predsb<0.5] = 0
 
    # Evaluating classification
    evaluationC = c_evaluate(yt, predsb)
    arrAcc.append(evaluationC[0])
    meanAcc.append(evaluationC[1])
    arrF1.append(evaluationC[2])
    meanF1.append(evaluationC[3])
    np.savetxt(evaluationl_path+'arrAcc.txt', arrAcc) 
    np.savetxt(evaluationl_path+'meanAcc.txt', meanAcc)
    np.savetxt(evaluationl_path+'arrF1.txt', arrF1) 
    np.savetxt(evaluationl_path+'meanF1.txt', meanF1)
    
#    calculate the cutting indexes     
    if confidence == 1:
        alpha = excludingPercentage
   
    elif confidence == 2:  
        x_1 = dataValues[initialDataLength-sizeOfBatch:initialDataLength,...]
        x_2 = Ut
        alpha = cuttingPercentage(x_1, x_2, t=t)[1]
    print('cuttingPercentage:', alpha)
    cuttingNumber = int(sizeOfBatch*alpha)
    idx_del = np.argpartition(W[0], cuttingNumber)[:cuttingNumber]
    
#    denoising    
    if denoising != 0:
        if denoising == 2:
            if len(meanCon) > 1:
                if meanCon[-1] - meanCon[-2] > 0:
                    denoisingProbability = denoisingProbability - 0.01
                else:
                    denoisingProbability = denoisingProbability + 0.01
                if denoisingProbability > 0.9:
                    denoisingProbability = 0.9
                elif denoisingProbability < 0.5:
                    denoisingProbability = 0.5
                arrDenPro.append(denoisingProbability)
                np.savetxt(evaluationl_path+'arrDenPro.txt', arrDenPro)
            print('denoisingProbability:', denoisingProbability)     
                
        predsb_d = predicted.copy()
        predsb_d[predsb_d>=denoisingProbability] = 1
        predsb_d[predsb_d<=(1-denoisingProbability)] = 0
        
        Ut = np.delete(Ut, idx_del, axis=0)
        predsb_d = np.delete(predsb_d, idx_del, axis=0)  
        
        size = sizeOfWindow - sizeOfBatch + cuttingNumber
        X = X[-size:]
        y = y[-size:]
        
        X, y = np.concatenate((X, Ut),axis=0), np.concatenate((y, predsb_d),axis=0)
        
        print('added predicted data set for next round training:', Ut.shape[0], predsb_d.shape[0])
    
    else:        
        Ut = np.delete(Ut, idx_del, axis=0)
        predsb = np.delete(predsb, idx_del, axis=0)  
        
        size = sizeOfWindow - sizeOfBatch + cuttingNumber
        X = X[-size:]
        y = y[-size:]
        
        X, y = np.concatenate((X, Ut),axis=0), np.concatenate((y, predsb),axis=0)
       
        print('added predicted data set for next round training:', Ut.shape[0], predsb.shape[0])
    
    if reset == True:
        clf = createCNN(outsize = outsize, input_shape = input_shape)
        clf.compile(loss = keras.losses.binary_crossentropy, optimizer= ada)
        
    end1 = time.time()
    updatingTime.append(end1-start1)
    np.savetxt(evaluationl_path+'updatingTime.txt', updatingTime)
