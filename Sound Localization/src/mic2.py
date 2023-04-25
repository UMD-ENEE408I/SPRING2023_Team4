import pyaudio
import wave
import array as arr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy as sp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# WIFI/SOCKET START
import socket # to connect to C++ robot code 
import struct
import time

ip = '' # Bind to all network interfaces
port = 3333
max_buffer_size  = 1024

fs = 44100 # samples per second
dt = 1/fs # for cross correlation
FORMAT = pyaudio.paInt16 # 16 bits per sample
#Audio Channels
CHANNELS = 2 # 2 microphones
chunk = 1024 # record in chunks of 1024 samples
filename = "output.wav"

#initialize variables used in case statements
theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
theta = 0 #initialize theta
maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta
state = 0 #State machine State 0 = initial spin
ip_address = 0; sig1 = 0; sig1_rms = 0; audio = 0; frames = 0
d = .15 #distance between 2 microphones (in meters) CHANGE
v = 343 #speed of sound /sec (20C through air)

#START GETTING VALUES OF THETA
while __name__ == '__main__':
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket.bind((ip, port))

    if state == 0: #Read theta values until they =2pi, meaning we having finished 360 degrees 
        #open audio streams for each mic
        audio = pyaudio.PyAudio() 
        # mic 1
        stream1 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = 0)
        # mic 2
        stream2 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = 1)

        frames1 = [] # stores recorded data of mic1
        frames2 = [] #stores recorded data of mic2
        sig1_rms = [] #stores rms values of signal 1
        sig2_rms = [] #rms of signal 2
        
        while(theta<2*np.pi): #record and receive packet until i=2pi (ie mouse spun 360)
        #RECEIVE PACKET
            (theta_by, ip_address) = UDPServerSocket.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
            theta_str = theta_by.decode('utf-8') #decode bytes to a string
            theta_split = theta_str.split() #split string into words by space (gets indiidual values)
            time_str = theta_split[1] #store ONLY time values
            theta_str = theta_split[0] #store ONLY theta value in theta_str WILL IT BE MORE THAN LENGTH 2??
            
            theta = float(theta_str) # convert string to float number
            t = float(time_str) #convert string to float
        
            #REMEMBER: theta_list[i][0] = 0, extended after this first value of 0
            theta_list[0].extend([t]) #extend with new time 
            theta_list[1].extend([theta]) #extend with new theta

            print('Message received: {}'.theta_str) #prints theta as string
 
            data1 = stream1.read(chunk)
            data2 = stream2.read(chunk)
            sig1 = np.frombuffer(sig1, dtype=np.int16) #convert signal to numpy array
            sig2 = np.frombuffer(sig2, dtype=np.int16)
            sig1 = sig1.astype(np.float64)
            sig2 = sig2.astype(np.float64)
            rms1 = np.sqrt(np.mean(sig1**2))
            rms2 = np.sqrt(np.mean(sig2**2))
            sig1_rms.append(rms1)
            sig2_rms.append(rms2)
            frames1.append(data1)
            frames2.append(data2)
            #i = theta


        # Stop and close the streams
        stream1.stop_stream()
        stream2.stop_stream()
        stream1.close()
        stream2.close()
        audio.terminate()
        sig1_rms = np.array(sig1_rms) #convert list to array so we can get index for peaks
        sig2_rms = np.array(sig2_rms)
        #Create Max Theta Array
        maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final max
        maxtheta[0] = 0 
        state =1 #NEXT STATE

    elif state == 1:
        f = 1 #fundamental frequency: what is f in this case, it is 1 in example
        #normalize signals
        sig1 = sig1_rms / np.linalg.norm(sig1_rms)  #new variable or this fine?
        sig2 = sig2_rms / np.linalg.norm(sig2_rms)
        C_sig1sig2 = np.correlate(sig1, sig2, mode='full') #Calculate correlation
        N = C_sig1sig2.shape[0] #length of C
        T = N*dt #length of signal (time)
        t_shift_C = np.arange(-T + dt, T, dt)#Calculate corresponding timeshift corresponding to each index HOW?
       
        C_norm_sig1 = np.zeros(C_sig1sig2.shape[0])
        C_norm_sig2 = np.zeros(C_sig1sig2.shape[0])

        center_index = int((C_sig1sig2.shape[0] + 1) / 2) - 1 #index of zero shift
        low_shift_index = int((C_sig1sig2.shape[0] + 1) / 2) + 1
        high_shift_index = int((C_sig1sig2.shape[0] + 1) / 2) - 1
        for i in range(low_shift_index, high_shift_index + 1):
            low_norm_index = max(0, i)
            high_norm_index = min(sig1.shape[0], i + sig1.shape[0])
            C_norm_sig1[i + center_index] = np.linalg.norm(sig1[low_norm_index:high_norm_index])

            low_norm_index = max(0, -i)
            high_norm_index = min(sig2.shape[0], -i + sig2.shape[0])
            C_norm_sig2[i + center_index] = np.linalg.norm(sig2[low_norm_index:high_norm_index])
        
        #Normalize calculated correlation per shift
        C_sig1sig2_normalized_per_shift = C_sig1sig2 / (C_norm_sig1 * C_norm_sig2)
       
        max_indices_back = -int(((1 / f) / 2) / dt) + center_index
        max_indices_forward = int(((1 / f) / 2) / dt) + center_index
        i_max_C_normalized = np.argmax(C_sig1sig2_normalized_per_shift[max_indices_back:max_indices_forward + 1]) + max_indices_back
        t_shift_hat_normalized = t_shift_C[i_max_C_normalized] #time where signals are most similar
        #Time difference of arrival
        TDoA = (i_max_C_normalized - center_index) * dt #index difference bw max correlation value and zero shift = time difference bw signals, *dt converts to time units
        #Angle of Arrival
        maxtheta[1] = np.arcsin(TDoA * v / d) #SHOULD get right angle, 
        
        spf = wave.open("output.wav", "r")
        fs = spf.getframerate()
        Time = np.linspace(0, len(sig1_rms) / fs, num=len(sig1_rms))
        #Filter Mic signal NORM OF LAST 100s
        # def normalize(x,axis = 0):
        #     return sklearn.preprocessing.minmax_scale(x,axis=axis)

        #SIGNALS MUST BE DOWNSAMPLED TO THE SAMPLING RATE OF THE UDP
        #mic sigs fs= 44100 Hz
        
        #sig1_norm = normalize(sig1)

        #When should we interpolate signals to the timing of UDP? before or after correlation?
        sig1_inp = interp1d(theta_list[0],sig1_rms, kind = 'nearest') #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
        state = 2

    elif (state == 2): #Send Packet   
        x = maxtheta[1]
        print('Sending {}'.format(x)) # print here what we are sending
        x = struct.pack("f", x[0]) #Send back theta
        UDPServerSocket.sendto(x, ip_address)
        state = 3

    #elif (state == 3):
        #Keep recording and correlating



