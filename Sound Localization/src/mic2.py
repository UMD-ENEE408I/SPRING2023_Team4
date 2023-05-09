import pyaudio
import wave
import array as arr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, lfilter
from scipy.interpolate import interp1d
#from sklearn.preprocessing import normalize
# WIFI/SOCKET START
import socket # to connect to C++ robot code 
import struct
import threading
import time

ip = '' # Bind to all network interfaces
#mouse1_port = 3333
mouse2_port = 3334 #change back to 3334
max_buffer_size  = 1024

fs = 44100 # samples per second
dt = 1/fs # for cross correlation
FORMAT = pyaudio.paInt16 # 16 bits per sample
#Audio Channels
CHANNELS = 1 #1 microphone
chunk = 1024 # record in chunks of 1024 samples
cutoff_freq = 1100
filter_order = 4 #Change as needed
lowcut = 900
highcut = 1100
#initialize variables used in case statements
mouse2_theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
mouse2_theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
d = .0826 #distance between 2 microphones (in meters) CHANGE
v = 343 #speed of sound /sec (20C through air)
i = 0 #initialize incrementing variable
mouse2_maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta
data1 = None; data2 = None
state = 0 #State machine State 0 = initial spin
diff = 10; mouse1_ip_address = 0; mouse2_ip_address = 0; sig1val = 0; sig2val = 0; sig1_rms = 0; sig2_rms = 0; sig3 = 0; sig3_rms = 0; audio = 0; frames3 = 0; mouse1_theta = 0; mouse2_theta = 0
nyq = .5*fs
count = 0
f = 1000 #fundamental frequency: 1kHz sound source
#filter

def butter_filter(data, lowcut, highcut, fs, filter_order):
    norm_lowcut = lowcut/nyq
    norm_highcut = highcut/nyq
    b,a = signal.butter(filter_order, [norm_lowcut, norm_highcut], btype = 'band')
    #print(data.shape)
    y = lfilter(b,a,data)
    return y

def readchunk(stream):
    data = stream.read(chunk)
    return data

def read_stream1():
    global data1
    data1 = readchunk(stream1)

def read_stream2():
    global data2
    data2 = readchunk(stream2)


while __name__ == '__main__':
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
                            input_device_index = id[1]) #NEED TO ACTUALLY CHECK WHICH INDEX CORRESPONDS TO WHICH MOUSE
        # mic 2
        stream2 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[3])

        
        frames1 = [] # stores recorded data of mic1
        frames2 = [] #stores recorded data of mic2
        frames3 = [] # stores recorded data
        sig1_rms_times = []
        sig2_rms_times = []
        sig3_rms_times = []
        #sig1_times = []
        sig1 = []
        sig2 = []
        sig1_rms = []
        sig2_rms = []
        sig3_rms = [] #stores rms values of signal
        mouse1_t_start = None
        mouse1_t_data_start = None
        mouse2_t_start = None
        mouse2_t_data_start = None

        

        while (mouse2_theta<2*np.pi): #ONLY BASED ON MOUSE 1 IS THIS OK? until theta equals 2pi #int(fs/ chunk * 5)): #change number w/ chunk * to change recording length
            #START RECORDING, WHEN WE START GETTING VALUES OF THETA
            #RECEIVE PACKET, will this do this continuously as the rest of the case is completed? 
     
            (mouse2_theta_by, mouse2_ip_address) = UDPServerSocket2.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
            mouse2_theta_str = mouse2_theta_by.decode('utf-8') #decode bytes to a string
            mouse2_theta_split = mouse2_theta_str.split() #split string into words by space (gets indiidual values)
            mouse2_time_str = mouse2_theta_split[1] #store ONLY time values
            mouse2_theta_str = mouse2_theta_split[0] #store ONLY theta value in theta_str WILL IT BE MORE THAN LENGTH 2??
            
            mouse2_theta = float(mouse2_theta_str) # convert string to float number
            print('Message received from Mouse 2: {}'.format(mouse2_theta)) #prints theta as string
    
            if mouse2_t_start is None:
                mouse2_t_start = float(mouse2_time_str) #convert string to float
                mouse2_t = 0
            else:
                mouse2_t = float(mouse2_time_str) - mouse2_t_start
            #print('time: {}'.format(t))
            #print('timestr: {}'.format(time_str)) #prints theta as string
            #print('thetastr: {}'.format(theta_str))

            #Record
#             data1 = stream1.read(chunk)
#             data2 = stream2.read(chunk)
#Threading
            thread1_read = threading.Thread(target = read_stream1)
            thread2_read = threading.Thread(target = read_stream2)
            thread1_read.start()
            thread2_read.start()
            thread1_read.join()
            thread2_read.join()
            
            if mouse2_t_data_start is None:
                mouse2_t_data_start = time.time()
                mouse2_t_data = 0
            else:
                mouse2_t_data =  time.time() - mouse2_t_data_start  
            
            sig1_rms_times.append(mouse2_t_data)
            sig2_rms_times.append(mouse2_t_data)
            #Record all 3 mics
         
            sig1val = np.frombuffer(data1, dtype=np.int16) 
            sig1val = sig1val.astype(np.float64)
            sig1.append(sig1val)
            
            #rms1 = np.sqrt(np.mean(sig1**2))
            #sig1_rms.append(rms1)
            frames1.append(data1)

            sig2val = np.frombuffer(data2, dtype=np.int16) 
            sig2val = sig2val.astype(np.float64)
            sig2.append(sig2val)
            
            #rms2 = np.sqrt(np.mean(sig2**2))
            #sig2_rms.append(rms2)
            frames2.append(data2)
            
            print('Message received from Mouse 2: {}'.format(mouse2_theta_str)) #prints theta as string
    
            
            #REMEMBER: theta_list[i][0] = 0, extended after this first value of 0
            mouse2_theta_list[0].extend([mouse2_t]) #extend with new time 
            mouse2_theta_list[1].extend([mouse2_theta]) #extend with new theta
    
        #END OF SPIN
        sig1 = np.array(sig1)
        sig2 = np.array(sig2)

        print('SIG1 SIZE')
        print(sig1.shape)
        filtered_sig1 = butter_filter(sig1,lowcut,highcut,fs,filter_order)
        filtered_sig2 = butter_filter(sig2,lowcut,highcut,fs,filter_order)
        for i in range(len(sig1_rms_times)):
            rms1 = np.sqrt(np.mean(filtered_sig1[i]**2))
            sig1_rms.append(rms1)
            rms2 = np.sqrt(np.mean(filtered_sig2[i]**2))
            sig2_rms.append(rms2)
        # Stop and close the stream
        stream1.stop_stream()
        stream2.stop_stream()
        stream1.close()
        stream2.close()
        #audio.terminate()
        data1 = None
        data2 = None
        sig1_rms = np.array(sig1_rms) #convert list to array so we can get index for peaks
        #print(sig1_rms.shape)
        sig2_rms = np.array(sig2_rms)
        mouse2_maxtheta[0] = 0 
        state = 2 #NEXT STATE

    elif state == 2: #find peaks/cross correlation to find maxtime/maxtheta
        #1 MIC
        # Open and plot output.wav
#         wf1 = wave.open("output1.wav", 'wb')
#         wf1.setnchannels(CHANNELS)
#         wf1.setsampwidth(audio.get_sample_size(FORMAT))
#         wf1.setframerate(fs)
#         wf1.writeframes(b''.join(frames1))
#         wf1.close()
#         spf1 = wave.open("output1.wav", "r")
#         
#         wf2 = wave.open("output2.wav", 'wb')
#         wf2.setnchannels(CHANNELS)
#         wf2.setsampwidth(audio.get_sample_size(FORMAT))
#         wf2.setframerate(fs)
#         wf2.writeframes(b''.join(frames2))
#         wf2.close()
#         spf2 = wave.open("output2.wav", "r")
#     
#         fs1 = spf1.getframerate()
        #Time = np.linspace(0, len(sig3_rms) / fs, num=len(sig3_rms))

        #interpolate to timing of mouse packet

        sig1_interpolater = interp1d(sig1_rms_times,sig1_rms, kind = 'nearest', bounds_error=False) #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
        sig2_interpolater = interp1d(sig2_rms_times,sig2_rms, kind = 'nearest', bounds_error=False)
        sig1_inp = sig1_interpolater(mouse2_theta_list[0])
        sig2_inp = sig2_interpolater(mouse2_theta_list[0])
        
        Time = np.linspace(0, len(mouse2_theta_list[0]) , num=len(sig1_inp))
        
#    PLOTTING
#         plt.figure(1)
#         plt.title("Signal Wave")
#         # plt.plot(Time, sig1) #Amplitude of frequency
#         plt.subplot(3,1,1)
#         plt.plot(Time, sig1_inp, color='r')
#         
#         plt.subplot(312)
#         plt.plot(Time, sig2_inp)
#         
#         plt.show(block = False)

        sig1_peaks, _ = find_peaks(sig1_inp, prominence = 1)
        sig2_peaks, _ = find_peaks(sig2_inp, prominence = 1)
        sig1_maxpeak = sig1_peaks[np.argmax(sig1_inp[sig1_peaks])]
        sig2_maxpeak = sig2_peaks[np.argmax(sig2_inp[sig2_peaks])]
        
        mean_maxpeak = (sig1_maxpeak + sig2_maxpeak) / 2
        
        mouse2_xmax = mouse2_theta_list[0][int(mean_maxpeak)]
        mouse2_maxtheta[1] = mouse2_theta_list[1][int(mean_maxpeak)]
        
  
        state = 3 # GO TO NEXT STATE

    elif state == 3: #Send packet with value of theta
        print('In state 3')

        x2 = mouse2_maxtheta[1]
        print('Sending {}'.format(x2)) # print here what we are sending
        x2 = struct.pack("f", x2) #Send back theta
        UDPServerSocket2.sendto(x2, mouse2_ip_address)
        
        if count <5: # REPEAT STEPS 5 TIMES
            count = count + 1
            state = 4
        else:
            print('exiting')
            exit()
    
        #state = 4 # Back to state 1 for another spin?
        
    elif state == 4: #Check that mouse is stopped, when stopped, record, stop recording when state of mouse changes
        #stream1.stop_stream()
        #stream2.stop_stream()
        #stream1.close()
        #stream2.close()
        #audio.terminate()
        #stream1 = None
        #stream2 = None
        stream1 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[1])
        stream2 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[3])

        frames1 = [] # stores recorded data of mic1
        frames2 = [] #stores recorded data of mic2
        sig1_rms = [] #stores rms values of signal 1
        sig2_rms = [] #rms of signal 2
        sig1 = []
        sig2 = []
        mouse2_state = 3
        
#         (mouse2_state_by, mouse2_ip_address) = UDPServerSocket2.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
#         mouse2_state_str = mouse2_state_by.decode('utf-8') #decode bytes to a string
#         mouse2_state = float(mouse2_state_str)
#         print('mousestate:{}'.format(mouse2_state))
        #exit()
        while(mouse2_state != 4):
            (mouse2_state_by, mouse2_ip_address) = UDPServerSocket2.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
            mouse2_state_str = mouse2_state_by.decode('utf-8') #decode bytes to a string
            mouse2_state = float(mouse2_state_str)
            print('mouse2_state = {}'.format(mouse2_state))
            
#             data1 = stream1.read(chunk)
#             data2 = stream2.read(chunk)
            
            thread1_read = threading.Thread(target = read_stream1)
            thread2_read = threading.Thread(target = read_stream2)
            thread1_read.start()
            thread2_read.start()
            thread1_read.join()
            thread2_read.join()
            
            sig1val = np.frombuffer(data1, dtype=np.int16) 
            sig1val = sig1val.astype(np.float64)
            sig1 = np.concatenate((sig1, sig1val)) #changed from append
            
            #rms1 = np.sqrt(np.mean(sig1**2))
            #sig1_rms.append(rms1)
            frames1.append(data1)

            sig2val = np.frombuffer(data2, dtype=np.int16) 
            sig2val = sig2val.astype(np.float64)
            sig2 = np.concatenate((sig2, sig2val)) #.append(sig2val) #changed from append
            
            #rms2 = np.sqrt(np.mean(sig2**2))
            #sig2_rms.append(rms2)
            frames2.append(data2)
            

            
            
            if(mouse2_state ==4):
                print('hereeee')
                stream1.stop_stream()
                stream2.stop_stream()
                stream1.close()
                stream2.close()
                state =5 # GOT TO NEXT STATE
                #stop recording when mouse leaves state 3:
                sig1 = np.array(sig1)
                sig2 = np.array(sig2)

                filtered_sig1 = butter_filter(sig1,lowcut,highcut,fs,filter_order)
                filtered_sig2 = butter_filter(sig2,lowcut,highcut,fs,filter_order)


        #stop recording when mouse leaves state 3:
#         stream1.stop_stream()
#         stream2.stop_stream()
#         stream1.close()
#         stream2.close()
#         audio.terminate()
#         sig1_rms = np.array(sig1_rms) #convert list to array so we can get index for peaks
#         sig2_rms = np.array(sig2_rms)
#         state = 5
        
    elif state == 5: #Cross correlate, find TDOA and angle, send back to mouse
         #normalize signals
        sig1 = filtered_sig1 / np.linalg.norm(filtered_sig1)  #new variable or this fine?
        sig2 = filtered_sig2 / np.linalg.norm(filtered_sig2)
        print(sig1.shape)
        print(sig2.shape)
        
        
#    PLOTTING
#         Time = np.linspace(0, len(sig1) , num=len(sig1))
#         plt.figure(1)
#         plt.title("Signal Wave")
#         # plt.plot(Time, sig1) #Amplitude of frequency
#         plt.subplot(3,1,1)
#         plt.plot(Time, sig1, color='r')
#         
#         plt.subplot(312)
#         plt.plot(Time, sig2)
#         
#         plt.show()
       
        C_sig1sig2 = sp.signal.correlate(sig1, sig2, mode='full') #Calculate correlation
        print(C_sig1sig2.shape)
        N = sig1.shape[0] #length of C
        T = N*dt #length of signal (time)
        t_shift_C = np.arange(-T + dt, T, dt)#Calculate corresponding timeshift corresponding to each index HOW?
       
        sig1_ones = np.ones((sig1.shape[0],))
        sig1_square = np.square(sig1)
        sig1_sum_square = sp.signal.correlate(sig1_square, sig1_ones, 'full')
        C_norm_sig1 = np.sqrt(sig1_sum_square)
        
        sig2_ones = np.ones((sig2.shape[0],))
        sig2_square = np.square(sig2)
        sig2_sum_square = sp.signal.correlate(sig2_square, sig2_ones, 'full')
        C_norm_sig2 = np.sqrt(sig2_sum_square)

        C_sig1sig2_normalized_per_shift = C_sig1sig2 / (C_norm_sig1 * C_norm_sig2)
       
        center_index = int((C_sig1sig2.shape[0] + 1) / 2) - 1

        max_indices_back = -int(((1 / f) / 2) / dt) + center_index
        max_indices_forward = int(((1 / f) / 2) / dt) + center_index
        i_max_C_normalized = np.argmax(C_sig1sig2_normalized_per_shift[max_indices_back:max_indices_forward + 1]) + max_indices_back
        t_shift_hat_normalized = t_shift_C[i_max_C_normalized] #time where signals are most similar
        #Time difference of arrival
        TDoA = (t_shift_hat_normalized) #index difference bw max correlation value and zero shift = time difference bw signals, *dt converts to time units
        print('TDOA={}'.format(TDoA))
        TDoA = abs(TDoA)
        #Angle of Arrival
        if TDoA>0: #if TDoA positive, positive maxtheta
            mouse2_maxtheta[1] = np.arctan(TDoA * v / d)  # subtract target_omega in mouse?
        elif TDoA<0:
            mouse2_maxtheta[1] = -1*np.arctan(TDoA * v / d)
        else:
            mouse2_maxtheta[1] = 0
        state = 3 #Go back to state 3: Send packet


#         center_index = int((C_sig1sig2.shape[0] -1)/2)#+ 1) / 2) - 1 #index of zero shift
#         low_shift_index = -center_index#int((C_sig1sig2.shape[0] + 1) / 2) + 1
#         high_shift_index = center_index#int((C_sig1sig2.shape[0] + 1) / 2) - 1
#         for i in range(low_shift_index, high_shift_index + 1):
#             low_norm_index = max(0, -i)
#             high_norm_index = min(sig1.shape[0] - 1, sig1.shape[0] - 1 - i)
#             C_norm_sig1[i + center_index] = np.linalg.norm(sig1[low_norm_index:high_norm_index])
#             
#             low_norm_index = max(0, i)
#             high_norm_index = min(sig2.shape[0] - 1, sig2.shape[0] - 1 + i)
#             C_norm_sig2[i + center_index] = np.linalg.norm(sig2[low_norm_index:high_norm_index])
#         #print(C_norm_sig1)
        
        #Normalize calculated correlation per shift
#         C_sig1sig2_normalized_per_shift = C_sig1sig2 / (C_norm_sig1 * C_norm_sig2)
#        
        
    else:
        print("HERE IN ELSE EXITING")
        exit()
    #once theta=2pi, exit loop -> spin is over2
    #END OF WHILE LOOP




