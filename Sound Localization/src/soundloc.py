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
mouse1_port = 3333
mouse2_port = 3334
max_buffer_size  = 1024

fs = 44100 # samples per second
dt = 1/fs # for cross correlation
FORMAT = pyaudio.paInt16 # 16 bits per sample
#Audio Channels
CHANNELS = 1 #1 microphone
#CHANNELS = 2 # 2 microphones
chunk = 1024 # record in chunks of 1024 samples
filename = "output.wav"

#initialize variables used in case statements
mouse1_theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
mouse1_theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
mouse2_theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
mouse2_theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
d = .15 #distance between 2 microphones (in meters) CHANGE
v = 343 #speed of sound /sec (20C through air)
i = 0 #initialize incrementing variable
mouse1_maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta
mouse2_maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta

state = 0 #State machine State 0 = initial spin
mouse1_ip_address = 0; mouse2_ip_address = 0; sig3 = 0; sig3_rms = 0; audio = 0; frames3 = 0; mouse1_theta = 0; mouse2_theta = 0
#START GETTING VALUES OF THETA
while __name__ == '__main__':
    UDPServerSocket1 = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket1.bind((ip, mouse1_port))
   
    UDPServerSocket2 = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket2.bind((ip, mouse2_port))
    
    if state == 0: #find indeces of microphones
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
        #find which microphone is on which mouse?
        state =1 #NEXT STATE

    elif state == 1: #initial spin, receive and record
        
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

        stream3 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input_device_index=id[2],
                            frames_per_buffer=chunk)
        
        frames1 = [] # stores recorded data of mic1
        frames2 = [] #stores recorded data of mic2
        frames3 = [] # stores recorded data
        sig1_rms = [] #stores rms values of signal 1
        sig2_rms = [] #rms of signal 2
        sig3_rms = [] #stores rms values of signal
    
        while (mouse1_theta<2*np.pi): #ONLY BASED ON MOUSE 1 IS THIS OK? until theta equals 2pi #int(fs/ chunk * 5)): #change number w/ chunk * to change recording length
            #START RECORDING, WHEN WE START GETTING VALUES OF THETA
            #RECEIVE PACKET, will this do this continuously as the rest of the case is completed? 
            (mouse1_theta_by, mouse1_ip_address) = UDPServerSocket1.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
            print(str(mouse1_theta_by))
            mouse1_theta_str = mouse1_theta_by.decode('utf-8') #decode bytes to a string
            mouse1_theta_split = mouse1_theta_str.split() #split string into words by space (gets indiidual values)
            mouse1_time_str = mouse1_theta_split[1] #store ONLY time values
            mouse1_theta_str = mouse1_theta_split[0] #store ONLY theta value in theta_str WILL IT BE MORE THAN LENGTH 2??
            
            mouse1_theta = float(mouse1_theta_str) # convert string to float number
            mouse1_t = float(mouse1_time_str) #convert string to float
        
            print('Message received from Mouse 1: {}'.format(mouse1_theta_str)) #prints theta as string
           
            (mouse2_theta_by, mouse2_ip_address) = UDPServerSocket2.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
            mouse2_theta_str = mouse2_theta_by.decode('utf-8') #decode bytes to a string
            mouse2_theta_split = mouse2_theta_str.split() #split string into words by space (gets indiidual values)
            mouse2_time_str = mouse2_theta_split[1] #store ONLY time values
            mouse2_theta_str = mouse2_theta_split[0] #store ONLY theta value in theta_str WILL IT BE MORE THAN LENGTH 2??
            
            mouse2_theta = float(mouse2_theta_str) # convert string to float number
            mouse2_t = float(mouse2_time_str) #convert string to float
        
            print('Message received from Mouse 2: {}'.format(mouse2_theta_str)) #prints theta as string
           
           
            #Record all 3 mics
            data1 = stream1.read(chunk)
            data2 = stream2.read(chunk)
            data3 = stream3.read(chunk)
            sig1 = np.frombuffer(data1, dtype=np.int16) #convert signal to numpy array
            sig2 = np.frombuffer(data2, dtype=np.int16)
            sig3 = np.frombuffer(data3, dtype=np.int16) # DO I NEED? converts signal to numpy array
            sig1 = sig1.astype(np.float64)
            sig2 = sig2.astype(np.float64)
            sig3 = sig3.astype(np.float64)
            #rms1 = np.sqrt(np.mean(sig1**2))
            #rms2 = np.sqrt(np.mean(sig2**2))
            rms3 = np.sqrt(np.mean(sig3**2))
            #sig1_rms.append(rms1)
            #sig2_rms.append(rms2)
            sig3_rms.append(rms3)
            frames1.append(data1)
            frames2.append(data2)
            frames3.append(data3)

            #REMEMBER: theta_list[i][0] = 0, extended after this first value of 0
            mouse1_theta_list[0].extend([mouse1_t]) #extend with new time 
            mouse1_theta_list[1].extend([mouse1_theta]) #extend with new theta
            mouse2_theta_list[0].extend([mouse2_t]) #extend with new time 
            mouse2_theta_list[1].extend([mouse2_theta]) #extend with new theta


        # Stop and close the stream
        stream1.stop_stream()
        stream2.stop_stream()
        stream3.stop_stream()
        stream1.close()
        stream2.close()
        stream3.close()
        audio.terminate()
        sig1 = np.array(sig1) #convert list to array so we can get index for peaks
        sig2 = np.array(sig2)
        sig3_rms = np.array(sig3_rms) #convert list to array so we can get index for peaks
        mouse2_maxtheta[0] = 0
        mouse1_maxtheta[0] = 0 
        state = 2 #NEXT STATE

    elif state == 2: #find peaks/cross correlation to find maxtime/maxtheta
        #1 MIC
        # Open and plot output.wave
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames3))
        wf.close()

        spf = wave.open("output.wav", "r")
        #sig1 = spf.readframes(-1)
        #sig1 = np.frombuffer(sig1, dtype=np.int16) # HERE OR ABOVE? converts signal to numpy array
        #sig1_rms = np.sqrt(np.mean(sig1**2))
        fs = spf.getframerate()
        #Time = np.linspace(0, len(sig3_rms) / fs, num=len(sig3_rms))
        #Filter Mic signal NORM OF LAST 100s
        # def normalize(x,axis = 0):
        #     return sklearn.preprocessing.minmax_scale(x,axis=axis)

        #SIGNALS MUST BE DOWNSAMPLED TO THE SAMPLING RATE OF THE UDP

        #This line has not been tested, the rest of getting signal, converting to volume, finding peaks WORKS
        sig3_inp = interp1d(mouse1_theta_list[0],sig3_rms, kind = 'nearest') #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
        Time = np.linspace(0, len(sig3_inp) / fs, num=len(sig3_inp))
        #FIND THETA FOR MOUSE 1
        #Plot mic 1 signal (not necessary)
        plt.figure(1)
        plt.title("Signal Wave")
        plt.plot(Time, sig3_inp, color='r')
        #plt.plot(Time, sig3_rms, color='r')

        #find max volume and time
        peaks, _ = find_peaks(sig3_inp,prominence=1) #try without prominence
        maxpeak = peaks[np.argmax(sig3_inp[peaks])]
        #peaks, _ = find_peaks(sig3_rms,prominence=1) #try without prominence
        #maxpeak = peaks[np.argmax(sig3_rms[peaks])]
        mouse1_xmax = Time[maxpeak] #time that max occurs
        #find element in theta_list that corresponds to time xmax
        for i in range(len(mouse1_theta_list)):
            if mouse1_xmax == mouse1_theta_list[0][i]: 
                mouse1_maxtheta[1] = mouse1_theta_list[1][i] #set maxtheta to the element in theta_list that occurs at time xmax
        print('Max peak of Mouse1 occurs at time= ', mouse1_xmax)
        plt.figure(2)
        plt.title("Max Peak")
        #plt.plot(Time,sig1) #Plot amplitude of frequency
        plt.plot(Time[peaks],sig3[peaks], 'x'); #plt.plot(signal); plt.legend(['prominence'])
        #plt.axvline(Time=xmax, ls='--', color="k")
        plt.show()


        #FIND THETA FOR MOUSE 2
        f = 1 #fundamental frequency: what is f in this case, it is 1 in example
        #normalize signals
        sig1 = sig1 / np.linalg.norm(sig1)  #new variable or this fine?
        sig2 = sig2 / np.linalg.norm(sig2)
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
        TDoA = t_shift_hat_normalized* dt #(i_max_C_normalized - center_index) * dt #index difference bw max correlation value and zero shift = time difference bw signals, *dt converts to time units
        #Angle of Arrival
        mouse2_maxtheta[1] = np.arcsin(TDoA * v / d) #SHOULD get right angle, hopefully
        
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
        sig1_inp = interp1d(mouse2_theta_list[0],sig1_rms, kind = 'nearest') #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
        

        state = 3 # GO TO NEXT STATE

    elif state == 3: #Send packet with value of theta
        print('In state 2')
            # SEND PACKET AFTER EXITING WHILE LOOP (SPIN IS OVER)
        x = mouse1_maxtheta[1]
        print('Sending {}'.format(x)) # print here what we are sending
        x = struct.pack("f", x[0]) #Send back theta
        UDPServerSocket1.sendto(x, mouse1_ip_address)
        state = 4

    elif state == 4: 
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



