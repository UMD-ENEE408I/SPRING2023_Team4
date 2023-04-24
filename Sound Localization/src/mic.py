import pyaudio
import wave
import array as arr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy as sp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
#from sklearn.preprocessing import normalize
# WIFI/SOCKET START
import socket # to connect to C++ robot code 
import struct
import time

ip = '' # Bind to all network interfaces
port = 3333
max_buffer_size  = 1024

fs = 44100 # samples per second
FORMAT = pyaudio.paInt16 # 16 bits per sample
#Audio Channels
CHANNELS = 1 #1 microphone
#CHANNELS = 2 # 2 microphones
chunk = 1024 # record in chunks of 1024 samples
filename = "output.wav"

#initialize variables used in case statements
theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
i = 0 #initialize incrementing variable
maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta
state = 0 #State machine State 0 = initial spin
ip_address = 0; sig1 = 0; sig1_rms = 0; audio = 0; frames = 0; theta = 0
#START GETTING VALUES OF THETA
while __name__ == '__main__':
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket.bind((ip, port))
    theta = 1 #set theta != 0 so we can enter while loop
    if state == 0:
        #initial spin, receive and record

        #START RECORDING, WHEN WE START GETTING VALUES OF THETA 
        #1 mic
        audio = pyaudio.PyAudio() 
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk)

        frames = [] # stores recorded data
        sig1_rms = [] #stores rms values of signal
        # Store data in chunks for 5 seconds
        while (theta<2*np.pi): #until theta equals 2pi #int(fs/ chunk * 5)): #change number w/ chunk * to change recording length

            #RECEIVE PACKET, will this do this continuously as the rest of the case is completed? 
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
            #Record
            data = stream.read(chunk)
            sig1 = np.frombuffer(sig1, dtype=np.int16) # DO I NEED? converts signal to numpy array
            sig1 = sig1.astype(np.float64)
            rms = np.sqrt(np.mean(sig1**2))
            sig1_rms.append(rms)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        sig1_rms = np.array(sig1_rms) #convert list to array so we can get index for peaks
        maxtheta[0] = 0 
        state = 1 #NEXT STATE
    elif state == 1: #Plot sound and find peaks and maxtime/maxtheta
        #1 MIC
        # Open and plot output.wave
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        spf = wave.open("output.wav", "r")
        #sig1 = spf.readframes(-1)
        #sig1 = np.frombuffer(sig1, dtype=np.int16) # HERE OR ABOVE? converts signal to numpy array
        #sig1_rms = np.sqrt(np.mean(sig1**2))
        fs = spf.getframerate()
        Time = np.linspace(0, len(sig1_rms) / fs, num=len(sig1_rms))
        #Filter Mic signal NORM OF LAST 100s
        # def normalize(x,axis = 0):
        #     return sklearn.preprocessing.minmax_scale(x,axis=axis)

        #SIGNALS MUST BE DOWNSAMPLED TO THE SAMPLING RATE OF THE UDP

        #This line has not been tested, the rest of getting signal, converting to volume, finding peaks WORKS
        sig1_inp = interp1d(theta_list[0],sig1_rms, kind = 'nearest') #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
        
        
        plt.figure(1)
        plt.title("Signal Wave")
        # plt.plot(Time, sig1) #Amplitude of frequency
        plt.plot(Time, sig1_rms, color='r')
        #plt.show()
        # Start animation
        peaks, _ = find_peaks(sig1_rms,prominence=1) #try without prominence
        maxpeak = peaks[np.argmax(sig1_rms[peaks])]
        xmax = Time[maxpeak] #time that max occurs
        #find element in theta_list that corresponds to time xmax
        for i in range(len(theta_list)):
            if xmax == theta_list[0][i]: 
                maxtheta[1] = theta_list[1][i] #set maxtheta to the element in theta_list that occurs at time xmax
    
        print('Max peak occurs at time= ', xmax)
        plt.figure(2)
        plt.title("Max Peak")
        #plt.plot(Time,sig1) #Plot amplitude of frequency
        plt.plot(Time[peaks],sig1[peaks], 'x'); #plt.plot(signal); plt.legend(['prominence'])
        #plt.axvline(Time=xmax, ls='--', color="k")
        plt.show()
        state = 2 # GO TO NEXT STATE

    elif state == 2: #Send packet with value of theta
        print('In state 2')
            # SEND PACKET AFTER EXITING WHILE LOOP (SPIN IS OVER)
        x = maxtheta[1]
        print('Sending {}'.format(x)) # print here what we are sending
        x = struct.pack("f", x[0]) #Send back theta
        UDPServerSocket.sendto(x, ip_address)
        state = 3

    elif state == 3: 
        print("State 3! Exit")
        exit()
    else:
        print("HERE IN ELSE EXITING")
        exit()
    #once theta=2pi, exit loop -> spin is over2
    #END OF WHILE LOOP




#The position of mouse at xmax is where we want it to rotate to for the starting position




#move on to other code (CROSS CORRELATION)

#begin recording of all 3 microphones
#store 3 signals as 3 different functions
#use cross correlation to compare functions, find time delay
#use time delay to triangulate/find angle

#2 MICS We will need to cross correlate signals to find the peak and the angle to send mouse back to



