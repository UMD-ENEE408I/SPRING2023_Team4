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
mouse1_theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
mouse1_theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
i = 0 #initialize incrementing variable
mouse1_maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta
state = 0 #State machine State 0 = initial spin
ip_address = 0; sig3 = 0; sig3_rms = 0; audio = 0; frames3 = 0; mouse1_theta = 0
count = 0
#START GETTING VALUES OF THETA
while __name__ == '__main__':
    UDPServerSocket1 = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket1.bind((ip, port))
    print('while loop')
    if state == 0:
        #initial spin, receive and record
        print('State 0')
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
#         stream3 = audio.open(format=FORMAT,
#                             channels=CHANNELS,
#                             rate=fs,
#                             input=True,
#                             frames_per_buffer=chunk)
        stream3 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[4])

        frames3 = [] # stores recorded data
        sig3_rms = [] #stores rms values of signal
        sig3_rms_times = []
        mouse1_t_start = None
        mouse1_t_data_start = None
        
        # Store data in chunks
        while (mouse1_theta<2*np.pi): #until theta equals 2pi #int(fs/ chunk * 5)): #change number w/ chunk * to change recording length
            print('here')
            (mouse1_theta_by, mouse1_ip_address) = UDPServerSocket1.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
            #print(str(mouse1_theta_by))
            mouse1_theta_str = mouse1_theta_by.decode('utf-8') #decode bytes to a string
            mouse1_theta_split = mouse1_theta_str.split() #split string into words by space (gets indiidual values)
            mouse1_time_str = mouse1_theta_split[1] #store ONLY time values
            mouse1_theta_str = mouse1_theta_split[0] #store ONLY theta value in theta_str WILL IT BE MORE THAN LENGTH 2??
            
            mouse1_theta = float(mouse1_theta_str) # convert string to float number
            
            if mouse1_t_start is None:
                mouse1_t_start = float(mouse1_time_str) #convert string to float
                mouse1_t = 0
            else:
                mouse1_t = float(mouse1_time_str) - mouse1_t_start
            #print('time: {}'.format(t))
            #print('timestr: {}'.format(time_str)) #prints theta as string
            #print('thetastr: {}'.format(theta_str))
            
            #Record
            data3 = stream3.read(chunk)
            if mouse1_t_data_start is None:
                mouse1_t_data_start = time.time()
                mouse1_t_data = 0
            else:
                mouse1_t_data =  time.time() - mouse1_t_data_start
                print('Message received from Mouse 1: {}'.format(mouse1_theta_str)) #prints theta as string
            
            sig3_rms_times.append(mouse1_t_data)
            sig3 = np.frombuffer(data3, dtype=np.int16) 
            sig3 = sig3.astype(np.float64)
            rms3 = np.sqrt(np.mean(sig3**2))
            sig3_rms.append(rms3)
            frames3.append(data3)

            #REMEMBER: theta_list[i][0] = 0, extended after this first value of 0
            mouse1_theta_list[0].extend([mouse1_t]) #extend with new time 
            mouse1_theta_list[1].extend([mouse1_theta]) #extend with new theta


        # Stop and close the stream
        stream3.stop_stream()
        stream3.close()
        #audio.terminate()
        sig3_rms = np.array(sig3_rms) #convert list to array so we can get index for peaks
        mouse1_maxtheta[0] = 0 
        state = 1 #NEXT STATE
    elif state == 1: #Plot sound and find peaks and maxtime/maxtheta
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
        #Filter Mic signal NORM OF LAST 100s
        # def normalize(x,axis = 0):
        #     return sklearn.preprocessing.minmax_scale(x,axis=axis)

        #SIGNALS MUST BE DOWNSAMPLED TO THE SAMPLING RATE OF THE UDP

        #This line has not been tested, the rest of getting signal, converting to volume, finding peaks WORKS
        sig3_interpolater = interp1d(sig3_rms_times, sig3_rms, kind = 'nearest', bounds_error=False) #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
        sig3_inp = sig3_interpolater(mouse1_theta_list[0])
        
        Time = np.linspace(0, len(mouse1_theta_list[0]) , num=len(sig3_inp))
        
#         
#         plt.figure(1)
#         plt.title("Signal Wave")
#         # plt.plot(Time, sig1) #Amplitude of frequency
#         plt.plot(Time, sig3_inp, color='r')
        #plt.show()
        # Start animation
        peaks, _ = find_peaks(sig3_inp,prominence=1) #try without prominence
        maxpeak = peaks[np.argmax(sig3_inp[peaks])]
        mouse1_xmax = mouse1_theta_list[0][maxpeak]#Time[maxpeak]  #time that max occurs
        #find element in theta_list that corresponds to time xmax
        for i in range(len(mouse1_theta_list[0])):
            
            if mouse1_xmax == mouse1_theta_list[0][i]:
                print('success!!!')
                mouse1_maxtheta[1] = mouse1_theta_list[1][i] #set maxtheta to the element in theta_list that occurs at time xmax
        
        print('Max peak occurs at time= ', mouse1_xmax)
#         plt.figure(2)
#         plt.title("Max Peak")
#         #plt.plot(Time,sig1) #Plot amplitude of frequency
#         plt.plot(Time[peaks],sig3_rms[peaks], 'x'); #plt.plot(signal); plt.legend(['prominence'])
#         #plt.axvline(Time=xmax, ls='--', color="k")
#         plt.show(block=False)
        x = mouse1_maxtheta[1]
        print('theta will be {}'.format(x))
        print('end of state 1')
        state = 2 # GO TO NEXT STATE
        print('go to state 2')

    elif state == 2: #Send packet with value of theta
        print('In state 2')
            # SEND PACKET AFTER EXITING WHILE LOOP (SPIN IS OVER)
        #x = maxtheta[1]
        print('Sending theta = {}'.format(x)) # print here what we are sending
        x = struct.pack("f", x) #Send back theta
        UDPServerSocket1.sendto(x, mouse1_ip_address)
        if count <5: # REPEAT STEPS 5 TIMES
            count = count + 1
            state = 3 # GO TO NEXT STATE
        else:
            print('exiting')
            exit()
    
        #state = 3

    elif state == 3: 
        stream3 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input = True,
                            frames_per_buffer=chunk)
        frames3 = [] # stores recorded data
        sig1_rms = []
        sig2_rms = []
        sig3_rms = []
        sig1 = []
        sig2 = []
        sig3 = []
        mouse1_state = 3
        mouse2_state = 3
        sig3_rms_times = []
        mouse1_t_start = None
        mouse1_t_data_start = None
        mouse1_theta = 0
        mouse1_track_theta = 0
        mouse1_theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
        mouse1_theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
        mouse1_maxtheta = arr.array('d', [0] * 2)
        
        while(mouse1_track_theta < 2*np.pi):
                (mouse1_theta_by, mouse1_ip_address) = UDPServerSocket1.recvfrom(max_buffer_size) 
                mouse1_theta_str = mouse1_theta_by.decode('utf-8') #decode bytes to a string
                mouse1_theta_split = mouse1_theta_str.split() #split string into words by space (gets indiidual values)
                mouse1_time_str = mouse1_theta_split[1] #store ONLY time values
                mouse1_theta_str = mouse1_theta_split[0] #store ONLY theta value in theta_str WILL IT BE MORE THAN LENGTH 2??
                mouse1_track_theta_str = mouse1_theta_split[2] #keep track of 0-2pi
                
                mouse1_theta = float(mouse1_theta_str) # convert string to float number
                mouse1_track_theta = float(mouse1_track_theta_str)
                if mouse1_t_start is None:
                    mouse1_t_start = float(mouse1_time_str) #convert string to float
                    mouse1_t = 0
                else:
                    mouse1_t = float(mouse1_time_str) - mouse1_t_start

                data3 = stream3.read(chunk)

                if mouse1_t_data_start is None:
                    mouse1_t_data_start = time.time()
                    mouse1_t_data = 0
                else:
                    mouse1_t_data =  time.time() - mouse1_t_data_start
                    print('Message received from Mouse 1: {}'.format(mouse1_theta_str)) #prints theta as string
                
                sig3_rms_times.append(mouse1_t_data)
                sig3 = np.frombuffer(data3, dtype=np.int16) 
                sig3 = sig3.astype(np.float64)
                rms3 = np.sqrt(np.mean(sig3**2))
                sig3_rms.append(rms3)
                frames3.append(data3)

                #REMEMBER: theta_list[i][0] = 0, extended after this first value of 0
                mouse1_theta_list[0].extend([mouse1_t]) #extend with new time 
                mouse1_theta_list[1].extend([mouse1_theta]) #extend with new theta
        sig3_inp = None
        sig3_interpolater = None
        sig3_peaks = None
        state = 4
        
    elif state == 4:

        
        sig3_interpolater = interp1d(sig3_rms_times,sig3_rms, kind = 'nearest', bounds_error=False) 
        sig3_inp = sig3_interpolater(mouse1_theta_list[0])

        Time3 = np.linspace(0, len(mouse1_theta_list[0]), num=len(sig3_inp))

        plt.figure(1)
        plt.title("Signal Wave")
        plt.plot(Time3, sig3_inp, color='r')

        sig3_peaks, _ = find_peaks(sig3_inp,prominence=1) #try without prominence
        sig3_maxpeak = sig3_peaks[np.argmax(sig3_inp[sig3_peaks])]
        #peaks, _ = find_peaks(sig3_rms,prominence=1) #try without prominence
        #maxpeak = peaks[np.argmax(sig3_rms[peaks])]
        mouse1_xmax = mouse1_theta_list[0][sig3_maxpeak] #time that max occurs
        mouse1_maxtheta[1] = mouse1_theta_list[1][sig3_maxpeak]
        #find element in theta_list that corresponds to time xmax
        # for i in range(len(mouse1_theta_list[0])):
        #     if mouse1_xmax == mouse1_theta_list[0][i]: 
        #         mouse1_maxtheta[1] = mouse1_theta_list[1][i] #set maxtheta to the element in theta_list that occurs at time xmax
        print('Max Theta Mouse 1 = {}'.format(mouse1_maxtheta[1]))
        x = mouse1_maxtheta[1]
        state = 2
        
    else:
        print("HERE IN ELSE EXITING")
        exit()
    #once theta=2pi, exit loop -> spin is over2
    #END OF WHILE LOOP

