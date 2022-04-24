# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 22:03:41 2022

@author: devin
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.interpolate import interp1d

measures = {}

def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def rolmean(dataset, hrw, fs):
    mov_avg = dataset['hart'].rolling(int(hrw*fs)).mean()
    avg_hr = (np.mean(dataset.hart))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*1.2 for x in mov_avg]
    dataset['hart_rollingmean'] = mov_avg

def detect_peaks(dataset):
    window = []
    peaklist = []
    listpos = 0
    for datapoint in dataset.hart:
        rollingmean = dataset.hart_rollingmean[listpos]
        if (datapoint < rollingmean) and (len(window) < 1):
            listpos += 1
        elif (datapoint > rollingmean):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
            listpos += 1
    measures['peaklist'] = peaklist
    measures['ybeat'] = [dataset.hart[x] for x in peaklist]

def calc_RR(dataset, fs):
    RR_list = []
    peaklist = measures['peaklist']
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt])
        ms_dist = ((RR_interval / fs) * 1000.0)
        RR_list.append(ms_dist)
        cnt += 1
    measures['RR_list'] = RR_list

def calc_bpm():
    RR_list = measures['RR_list']
    measures['bpm'] = 60000 / np.mean(RR_list)

def plotter(dataset, title):
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    plt.title(title)
    plt.plot(dataset.hart, alpha=0.5, color='blue', label="raw signal")
    plt.plot(dataset.hart_rollingmean, color ='green', label="moving average")
    plt.scatter(peaklist, ybeat, color='red', label="average: %.1f BPM" %measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.show()

def process(dataset, hrw, fs): #Remember; hrw was the one-sided window size (we used 0.75) and fs was the sample rate (file is recorded at 100Hz)
    rolmean(dataset, hrw, fs)
    detect_peaks(dataset)
    calc_RR(dataset, fs)
    calc_bpm()
    plotter(dataset, "My Heartbeat Plot")

peaklist = [63,
 165,
 264,
 360,
 460,
 565,
 674,
 773,
 863,
 953,
 1048,
 1156,
 1272,
 1385,
 1487,
 1592,
 1698,
 1803,
 1897,
 1994,
 2097,
 2206,
 2308,
 2406]

RR_list = [1020.0,
 990.0,
 960.0,
 1000.0,
 1050.0,
 1090.0,
 990.0,
 900.0,
 900.0,
 950.0,
 1080.0,
 1160.0,
 1130.0,
 1020.0,
 1050.0,
 1060.0,
 1050.0,
 940.0,
 970.0,
 1030.0,
 1090.0,
 1020.0,
 980.0]
RR_x = peaklist[1:] #Remove the first entry, because first interval is assigned to the second beat.
RR_y = RR_list #Y-values are equal to interval lengths

#Create evenly spaced timeline starting at the second peak, its endpoint and length equal to position of last peak
RR_x_new = np.linspace(RR_x[0],RR_x[-1],RR_x[-1]) 
f = interp1d(RR_x, RR_y, kind='cubic') #Interpolate the signal with cubic spline interpolation
print( f(250))
#Returns 997.619845418, the Y value at x=250

plt.title("Original and Interpolated Signal")
plt.plot(RR_x, RR_y, label="Original", color='blue')
plt.plot(RR_x_new, f(RR_x_new), label="Interpolated", color='red')
plt.legend()
plt.show()

dataset = pd.read_csv("data.csv")

#Calculate moving average with 0.75s in both directions, then append do dataset
hrw = 0.75 #One-sided window size, as proportion of the sampling frequency
fs = 100 #The example dataset was recorded at 100Hz
#Set variables
n = len(dataset.hart) #Length of the signal
n

frq = np.fft.fftfreq(len(dataset.hart), d=((1/fs))) #divide the bins into frequency categories
frq = frq[range(int(n/2))] #Get single side of the frequency range

#Do FFT
Y = np.fft.fft(f(RR_x_new))/n #Calculate FFT
Y = Y[range(int(n/2))] #Return one side of the FFT

#Plot
plt.title("Frequency Spectrum of Heart Rate Variability")
plt.xlim(0,0.6) #Limit X axis to frequencies of interest (0-0.6Hz for visibility, we are interested in 0.04-0.5)
plt.ylim(0, 50) #Limit Y axis for visibility
plt.plot(frq, abs(Y)) #Plot it
plt.xlabel("Frequencies in Hz")
plt.show()

lf = np.trapz(abs(Y[(frq>=0.04) & (frq<=0.15)])) #Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the area
print("LF:", lf)

hf = np.trapz(abs(Y[(frq>=0.16) & (frq<=0.5)])) #Do the same for 0.16-0.5Hz (HF)
print("HF:", hf)