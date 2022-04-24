# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:06:01 2022

@author: devin
"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
%matplotlib inline

freq1=1 #1Hz
time_step = 0.02
period1 = 1/freq1
time_vec = np.arange(0, 20, time_step)

sig1=(np.sin(2*np.pi/period1 * time_vec))
plt.figure(figsize=(60,10))
plt.plot(time_vec,sig1, label='low Frequency')

freq2=12 #12Hz
period2 = 1/freq2
noise_amplitude = 0.5
sig2 = noise_amplitude*(np.sin(2*np.pi/period2*time_vec))
plt.figure(figsize=(60,10))
plt.plot(time_vec, sig2, label='High Frequency')

freq3=7 #7Hz
period2 = 1/freq3
noise_amplitude = 0.5
sig3 = noise_amplitude*(np.sin(2*np.pi/period2*time_vec))
plt.figure(figsize=(60,10))
plt.plot(time_vec, sig3, label='Mid Frequency')

sig= sig1 + sig2 + sig3
plt.figure(figsize=(60,10))
plt.plot(time_vec, sig, label= 'Low,Mid, and High Frequency')

##Comput and plto the power##

#The FFT of the signal
sig_fft= fftpack.fft(sig)

#The power (sig_fft is of complex dtype)
power = np.abs(sig_fft)**2

#the corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

#plot the FFT power
plt.figure(figsize=(30, 20))
plt.plot(sample_freq, power)
plt.xlabel('Frequency[Hz]')
plt.ylabel('power')

#Find the peak frequency
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]
peak_freq

#Remove Frequencies
high_freq_fft=sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq]=0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(60,10))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.legend(loc='best')

##Check for frequencies being filtered out
#fft of the signal
sig_fft1=fftpack.fft(filtered_sig)

#the power of the signal
power=np.abs(sig_fft1)**2

#corresponding frequencies
sample_freq = fftpack.fftfreq(filtered_sig.size, d=time_step)

#Plotting FFT power
plt.figure(figsize=(30, 20))
plt.plot(sample_freq,power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('power')