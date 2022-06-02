# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:26:17 2022

@author:Byrb23
"""

import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
import skued
import pywt
import wfdb
from wfdb import processing
from detecta import detect_peaks
from ecgdetectors import Detectors
import csv


sample_rate = 360

def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def load_visualise(data_file):
    '''
    loads data and annotations, plots them 
    and returns data and annotations
    '''
    record = wfdb.rdrecord(data_file)
    sig, fields = wfdb.rdsamp(data_file, channels=[0])
    ann_ref=wfdb.rdann(data_file,'atr')
    p_signal=record.p_signal
    #load the signal
    ecg=p_signal[:,1]
    #get the annotations
    annotations = ann_ref
    return sig,fields,ecg, annotations

def ValSUREThresh(X):
    """
    Adaptive Threshold Selection Using Principle of SURE

    based from MATLAB

    """
    n = np.size(X)
    a = np.sort(np.abs(X))**2
    c = np.linspace(n-1,0,n)
    s = np.cumsum(a)+c*a
    risk = (n - (2 * np.arange(n)) + s)/n
   
    ibest = np.argmin(risk)
    THR = np.sqrt(a[ibest])
    return THR


def filter_and_visualise(data, sample_rate,wavelet,maxlev,thmethod):
    '''
    function that filters the signal using DTCWT 
    and visualises result
    '''
    forward2=skued.dtcwt(data,wavelet,'qshift1',mode='constant',level=maxlev,axis=- 1)
    for i in range(1, len(forward2)):     #Apply thresholding process to each decomposition level
        thr=ValSUREThresh(forward2[i])   
        forward2[i] = pywt.threshold(forward2[i], thr,mode=thmethod) 
    # First parameter is the object , Third  Parameter is the thresholding method (Hard or soft)   
    # Reconstruct signal  (iDT-DTCWT)    
    filtered=skued.idtcwt(forward2,wavelet,'qshift1', mode='constant', axis=- 1)
    #plt.figure(figsize=(12,3))
    #plt.title('Denoised')
    #plt.plot(filtered)
    #plt.show()
   
    return filtered


# Function borrowed from https://github.com/PathofData

def getrealpeaks(data_file,Peaks):
    Annotations= wfdb.rdann(data_file,'atr', return_label_elements=['symbol'])
    #RealPeaks= wfdb.Annotation(record_name='100', extension='atr', symbol=['N', 'V', 'A'])
    PeakSamples=Annotations.sample
    PeakSymbols=Annotations.symbol
    RealPeaksIndex=[]
    for index, sym in enumerate(PeakSymbols):
        if sym == 'N' or sym == 'V' or sym=='A':
            RealPeaksIndex=np.append(RealPeaksIndex, index)
    RealPeaksIndex= RealPeaksIndex.astype(int)
    RealPeaks=PeakSamples[RealPeaksIndex]
    TruePositive=[]
    FalsePositive=[]
    HitR=np.ones(len(RealPeaks), dtype= bool)
    for indP, ValP in np.ndenumerate(Peaks):
        Hit=0
        for indR, ValR in np.ndenumerate(RealPeaks):
            if np.absolute(ValP-ValR) < 50:
                Hit=1
                HitR[indR[0]]=False
                TruePositive=np.append(TruePositive, indP[0])
                RealPeaks= RealPeaks[HitR]
                HitR=HitR[HitR]
                break
        if Hit==0:
            FalsePositive=np.append(FalsePositive, indP[0])
    FalseNegative = len(HitR)
    TruePositiveRate=len(TruePositive)/(len(TruePositive)+len(HitR))
    PositivePredictiveValue=len(TruePositive)/(len(TruePositive)+len(FalsePositive))
    Sensitivity=len(TruePositive)/(len(TruePositive)+FalseNegative)
    Accuracy=len(FalsePositive)/(len(TruePositive)+len(FalsePositive)+FalseNegative)
    print('True Positive Count: {0:5d}'.format(len(TruePositive)))
    print('False Positive Count: {0:d}'.format(len(FalsePositive)))
    print('False Negative Count: {0:d}'.format(len(HitR)))
    print('TPR: {0:.3f}%'.format(TruePositiveRate*100))
    print('PPV: {0:.3f}%'.format(PositivePredictiveValue*100))
    print('Sensitivity: {0:.3f}%'.format(Sensitivity*100))
    print('Accuracy: {0:.3f}%'.format(Accuracy*100))     
    return 


#Load ECG Record
sig,fields,ecg,annotations=load_visualise('118e24')
wavelet='coif6'
maxlev=5
thmethod='hard'

filtered_ecg=filter_and_visualise(ecg, sample_rate,wavelet,maxlev,thmethod)
#HeartPy
#working_data, measures = hp.process(filtered_ecg,360)
#r_peaks=working_data['peaklist']
#rpeaks=np.array(r_peaks)
#Pan-Tompkins Detector
detectors = Detectors(sample_rate)
rpeaks =detectors.pan_tompkins_detector(filtered_ecg)#Get index of rpeaks
rpeaks=np.array(rpeaks) #Transform to numpy array to be passed on the comparitor function
comparitor = processing.Comparitor(annotations.sample[1:],
                                      rpeaks,
                                      int(0.1 * fields['fs']),
                                       ecg)              
comparitor.compare()
comparitor.print_summary()
#comparitor.plot()
accuracy=(comparitor.tp/(comparitor.fn+comparitor.tp+comparitor.fp) )* 100
sensitivity=(comparitor.tp/(comparitor.tp+comparitor.fn)) * 100 



print ("\nsignaltonoise ratio for datarec : ", 
        (signaltonoise_dB(filtered_ecg, axis = 0, ddof = 0)))
MSE = np.square(np.subtract(ecg,filtered_ecg)).mean()
print(" MSE : ",(MSE))
print (" accuracy : ", 
        (accuracy))
print (" Sensitivity : ", 
        (sensitivity))

#Write data to CSV 
#data_file='Record118e24'
#with open('mycsv.csv','a',newline='') as f:
#    writer=csv.writer(f)
#    writer.writerow([data_file,wavelet,maxlev,thmethod,signaltonoise_dB(filtered_ecg, axis = 0, ddof = 0),MSE ,accuracy,sensitivity])


