# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:10:52 2023

@author: bakam
"""
import pyaudio
import wave
import array as arr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy as sp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

FORMAT = pyaudio.paInt16 # 16 bits per sample
fs = 44100 # samples per second
#Audio Channels
CHANNELS = 1 #1 microphone
#CHANNELS = 2 # 2 microphones
chunk = 1024 # record in chunks of 1024 samples
filename = "output.wav"
state = 0 #State machine State 0 = initial spin

while state == 0: #find indeces of microphones #find device id for microphone
        audio = pyaudio.PyAudio() 
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount') #should be 3
        #j = 0
        id = []
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                id.append(i) #save device ids
                #j = j + 1
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i))
                print('\n')
        #find which microphone is on which mouse?
        print('DOne with Identifying Microphones ID')
        # The receivers of all 3 microphones have to be tested in order for this portion of the test to pass.
        state =1 #NEXT STATE

if state == 1: #initial spin, receive and record
        print ("IN STATE 1")
        stream1 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[0]) #NEED TO ACTUALLY CHECK WHICH INDEX CORRESPONDS TO WHICH MOUSE
        # mic 2
        stream2 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[1])

# =============================================================================
#         stream3 = audio.open(format=FORMAT,
#                             channels=CHANNELS,
#                             rate=fs,
#                             input_device_index=id[2],
#                             frames_per_buffer=chunk)
# =============================================================================
        
        frames1 = [] # stores recorded data of mic1
        frames2 = [] #stores recorded data of mic2
        frames3 = [] # stores recorded data
        sig1_rms = [] #stores rms values of signal 1
        sig2_rms = [] #rms of signal 2
        sig3_rms = [] #stores rms values of signal