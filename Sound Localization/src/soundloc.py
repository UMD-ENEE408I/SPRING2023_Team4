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
import socket 
import struct
import threading
import time


ip = '' # Bind to all network interfaces
mouse1_port = 3333
mouse2_port = 3334
max_buffer_size  = 1024

fs = 44100 # samples per second
dt = 1/fs # for cross correlation
FORMAT = pyaudio.paInt16 # 16 bits per sample
CHANNELS = 1 
chunk = 1024 # record in chunks of 1024 samples
cutoff_freq = 1100
filter_order = 4 #Change as needed
lowcut = 900
highcut = 1100

#initialize variables used in case statements
mouse1_theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
mouse1_theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
mouse2_theta_arr = arr. array('d', [0] * 100) # LIST array to store all values of theta so we can go back and pick theta corresponding the max amp
mouse2_theta_list =[[0],[0]] #[time],[theta]: list to store all values of theta sent over AND the time that they occurred at
d = .0698 #distance between 2 microphones (in meters) CHANGE
v = 343 #speed of sound /sec (20C through air)
i = 0
mouse1_maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta
mouse2_maxtheta = arr.array('d', [0] * 2) #only need two elements, initial 0 and final maxtheta
data1 = None; data2 = None
state = 0 #State machine State 0 = initial spin
mouse1_ip_address = 0; mouse2_ip_address = 0; sig1val = 0; sig2val = 0; 
sig1_rms = 0; sig2_rms = 0; sig3 = 0; sig3_rms = 0; audio = 0; frames3 = 0; 
mouse1_theta = 0; mouse2_theta = 0
nyq = .5*fs; count = 0; f = 1000 #fundamental freq
#START GETTING VALUES OF THETA

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

def mouse1_findtheta():
    sig3_inp = None
    sig3_interpolater = None
    sig3_interpolater = interp1d(sig3_rms_times,sig3_rms, kind = 'nearest', bounds_error=False) 
    sig3_inp = sig3_interpolater(mouse1_theta_list[0])

    #Time3 = np.linspace(0, len(mouse1_theta_list[0]), num=len(sig3_inp))
    # plt.figure(1)
    # plt.title("Signal Wave")
    # plt.plot(Time3, sig3_inp, color='r')

    sig3_peaks, _ = find_peaks(sig3_inp,prominence=1) #try without prominence
    sig3_maxpeak = sig3_peaks[np.argmax(sig3_inp[sig3_peaks])]

    mouse1_xmax = mouse1_theta_list[0][sig3_maxpeak] #time that max occurs
    mouse1_maxtheta[1] = mouse1_theta_list[1][sig3_maxpeak]
    #find element in theta_list that corresponds to time xmax
    # for i in range(len(mouse1_theta_list[0])):
    #     if mouse1_xmax == mouse1_theta_list[0][i]: 
    #         mouse1_maxtheta[1] = mouse1_theta_list[1][i] #set maxtheta to the element in theta_list that occurs at time xmax
    print('Max Theta Mouse 1 = {}'.format(mouse1_maxtheta[1]))
    # plt.figure(2)
    # plt.title("Max Peak")
    # #plt.plot(Time,sig1) #Plot amplitude of frequency
    # plt.plot(Time3[peaks],sig3_rms[peaks], 'x'); #plt.plot(signal); plt.legend(['prominence'])
    # #plt.axvline(Time=xmax, ls='--', color="k")
    # plt.show(block=False)


while __name__ == '__main__':
    UDPServerSocket1 = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket1.bind((ip, mouse1_port))
   
    UDPServerSocket2 = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket2.bind((ip, mouse2_port))
    
    if state == 0: #find indeces of microphones
        audio = pyaudio.PyAudio() 
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount') #should be 3
        id = []
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                id.append(i) #save device ids
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i))
        state =1 # GO TO NEXT STATE

    elif state == 1: #initial spin, receive and record
        stream1 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[1]) # Change depending on ids
        # mic 2
        stream2 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[3])

        stream3 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=chunk,
                            input_device_index = id[4])
        
        frames1 = [] # stores recorded data of mic1
        frames2 = [] #stores recorded data of mic2
        frames3 = [] # stores recorded data
        sig1_rms_times = []
        sig2_rms_times = []
        sig3_rms_times = []
        sig1 = []
        sig2 = []
        sig1_rms = []
        sig2_rms = []
        sig3_rms = [] #stores rms values of signal
        mouse1_t_start = None
        mouse1_t_data_start = None
        mouse2_t_start = None
        mouse2_t_data_start = None

        def mouse1_spin():
            while (mouse1_theta<2*np.pi): #ONLY BASED ON MOUSE 1 IS THIS OK? until theta equals 2pi #int(fs/ chunk * 5)): #change number w/ chunk * to change recording length
                #START RECORDING, WHEN WE START GETTING VALUES OF THETA
                #RECEIVE PACKET, will this do this continuously as the rest of the case is completed? 
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

        def mouse2_spin():
            while (mouse2_theta<2*np.pi): #START RECORDING AND RECEIVING PACKET, WHEN WE START GETTING VALUES OF THETA
                (mouse2_theta_by, mouse2_ip_address) = UDPServerSocket2.recvfrom(max_buffer_size) #WILL THETA BE IN BYTES, RIGHT? 
                mouse2_theta_str = mouse2_theta_by.decode('utf-8') #decode bytes to a string
                mouse2_theta_split = mouse2_theta_str.split() #split string into words by space (gets indiidual values)
                mouse2_time_str = mouse2_theta_split[1] #store ONLY time values
                mouse2_theta_str = mouse2_theta_split[0] #store ONLY theta value in theta_str WILL IT BE MORE THAN LENGTH 2??
                
                mouse2_theta = float(mouse2_theta_str) # convert string to float number
                print('Message from Mouse 2: {}'.format(mouse2_theta)) #prints theta as string
    
                if mouse2_t_start is None:
                    mouse2_t_start = float(mouse2_time_str) #convert string to float
                    mouse2_t = 0
                else:
                    mouse2_t = float(mouse2_time_str) - mouse2_t_start

                #Record
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
             
                sig1val = np.frombuffer(data1, dtype=np.int16) 
                sig1val = sig1val.astype(np.float64)
                sig1.append(sig1val)
                frames1.append(data1)

                sig2val = np.frombuffer(data2, dtype=np.int16) 
                sig2val = sig2val.astype(np.float64)
                sig2.append(sig2val)
                frames2.append(data2)
                #print('Message received from Mouse 2: {}'.format(mouse2_theta_str)) #prints theta as string

                #REMEMBER: theta_list[i][0] = 0, extended after this first value of 0
                mouse2_theta_list[0].extend([mouse2_t]) #extend with new time 
                mouse2_theta_list[1].extend([mouse2_theta]) #extend with new theta
        
        thread1_state1 = threading.Thread(target = mouse1_spin)
        thread2_state1 = threading.Thread(target= mouse2_spin)
        thread1_state1.start()
        thread2_state1.start()
        thread1_state1.join()
        thread2_state1.join()

        stream1.stop_stream()
        stream2.stop_stream()
        stream3.stop_stream()
        stream1.close()
        stream2.close()
        stream3.close()

        sig1 = np.array(sig1)
        sig2 = np.array(sig2)
        filtered_sig1 = butter_filter(sig1,lowcut,highcut,fs,filter_order)
        filtered_sig2 = butter_filter(sig2,lowcut,highcut,fs,filter_order)
        
        for i in range(len(sig1_rms_times)):
            rms1 = np.sqrt(np.mean(filtered_sig1[i]**2))
            sig1_rms.append(rms1)
            rms2 = np.sqrt(np.mean(filtered_sig2[i]**2))
            sig2_rms.append(rms2)

        data1 = None
        data2 = None
        sig1_rms = np.array(sig1_rms) #convert list to array so we can get index for peaks
        sig2_rms = np.array(sig2_rms)
        sig3_rms = np.array(sig3_rms) #convert list to array so we can get index for peaks
        mouse2_maxtheta[0] = 0
        mouse1_maxtheta[0] = 0 
        state = 2 # GO TO NEXT STATE

    elif state == 2: #find peaks/cross correlation to find maxtime/maxtheta
            # Open and plot output.wave
            # wf = wave.open("output3.wav", 'wb')
            # wf.setnchannels(CHANNELS)
            # wf.setsampwidth(audio.get_sample_size(FORMAT))
            # wf.setframerate(fs)
            # wf.writeframes(b''.join(frames3))
            # wf.close()

            # spf3 = wave.open("output3.wav", "r")

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
                    #fs = spf3.getframerate()
            #Time = np.linspace(0, len(sig3_rms) / fs, num=len(sig3_rms))
        # def mouse1_findtheta():
        #     sig3_interpolater = interp1d(sig3_rms_times,sig3_rms, kind = 'nearest', bounds_error=False) 
        #     sig3_inp = sig3_interpolater(mouse1_theta_list[0])

        #     #Time3 = np.linspace(0, len(mouse1_theta_list[0]), num=len(sig3_inp))

        #     # plt.figure(1)
        #     # plt.title("Signal Wave")
        #     # plt.plot(Time3, sig3_inp, color='r')

        #     sig3_peaks, _ = find_peaks(sig3_inp,prominence=1) #try without prominence
        #     sig3_maxpeak = sig3_peaks[np.argmax(sig3_inp[sig3_peaks])]
        #     #peaks, _ = find_peaks(sig3_rms,prominence=1) #try without prominence
        #     #maxpeak = peaks[np.argmax(sig3_rms[peaks])]
        #     mouse1_xmax = mouse1_theta_list[0][sig3_maxpeak] #time that max occurs
        #     mouse1_maxtheta[1] = mouse1_theta_list[1][sig3_maxpeak]
        #     #find element in theta_list that corresponds to time xmax
        #     # for i in range(len(mouse1_theta_list[0])):
        #     #     if mouse1_xmax == mouse1_theta_list[0][i]: 
        #     #         mouse1_maxtheta[1] = mouse1_theta_list[1][i] #set maxtheta to the element in theta_list that occurs at time xmax
        #     print('Max Theta Mouse 1 = {}'.format(mouse1_maxtheta[1]))
        #     # plt.figure(2)
        #     # plt.title("Max Peak")
        #     # #plt.plot(Time,sig1) #Plot amplitude of frequency
        #     # plt.plot(Time3[peaks],sig3_rms[peaks], 'x'); #plt.plot(signal); plt.legend(['prominence'])
        #     # #plt.axvline(Time=xmax, ls='--', color="k")
        #     # plt.show(block=False)


        #FIND THETA FOR MOUSE 2
        def mouse2_findtheta():
            f = 1 #fundamental frequency: what is f in this case, it is 1 in example
            #interpolate to timing of mouse packet
            sig1_interpolater = interp1d(sig1_rms_times,sig1_rms, kind = 'nearest', bounds_error = False) #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
            sig2_interpolater = interp1d(sig2_rms_times,sig2_rms, kind = 'nearest', bounds_error = False)
            sig1_inp = sig1_interpolater(mouse2_theta_list[0])
            sig2_inp = sig2_interpolater(mouse2_theta_list[0])

            sig1_peaks, _ = find_peaks(sig1_inp, prominence = 1)
            sig2_peaks, _ = find_peaks(sig2_inp, prominence = 1)
            sig1_maxpeak = sig1_peaks[np.argmax(sig1_inp[sig1_peaks])]
            sig2_maxpeak = sig2_peaks[np.argmax(sig2_inp[sig2_peaks])]
            
            mean_maxpeak = (sig1_maxpeak + sig2_maxpeak) / 2
            
            mouse2_xmax = mouse2_theta_list[0][int(mean_maxpeak)]
            mouse2_maxtheta[1] = mouse2_theta_list[1][int(mean_maxpeak)]

            # for i in range(len(mouse2_theta_list[0])): #find time where signals are most similar
            #     temp = sig1_inp[i] - sig2_inp[i]
            #     if(abs(temp) < diff): # find where sig difference is closest to zero
            #         diff = temp 
            #         #mouse2_xmax = mouse2_theta_list[0][i]
            #         mouse2_maxtheta[1] = mouse2_theta_list[1][i] #theta where they are most similar

        thread1_state2 = threading.Thread(target = mouse1_findtheta)
        thread2_state2 = threading.Thread(target = mouse2_findtheta)
        thread1_state2.start()
        thread2_state2.start()
        thread1_state2.join()
        thread2_state2.join()    

        state = 3 # GO TO NEXT STATE

    elif state == 3: #Send packet with value of theta
        print('In state 3')
            # SEND PACKET AFTER EXITING WHILE LOOP (SPIN IS OVER)
        def mouse1_sendtheta():
            x1 = mouse1_maxtheta[1]
            print('Sending {}'.format(x1)) # print here what we are sending
            x1 = struct.pack("f", x1) #Send back theta
            UDPServerSocket1.sendto(x1, mouse1_ip_address)

        def mouse2_sendtheta():
            x2 = mouse2_maxtheta[1]
            print('Sending {}'.format(x2)) # print here what we are sending
            x2 = struct.pack("f", x2) #Send back theta
            UDPServerSocket2.sendto(x2, mouse2_ip_address)

        thread1_state3 = threading.Thread(target = mouse1_sendtheta)
        thread2_state3 = threading.Thread(target = mouse2_sendtheta)
        thread1_state3.start()
        thread2_state3.start()
        thread1_state3.join()
        thread2_state3.join()    
        if count <5: # REPEAT STEPS 5 TIMES
            count = count + 1
            state = 4 # GO TO NEXT STATE
        else:
            print('exiting')
            exit()
    
        #state = 4 

    elif state == 4: # Receive packets and Record new signals
        print("State 4")
            
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
        # mic 3
        stream3 = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input = True,
                            frames_per_buffer=chunk,
                            input_device_index=id[2])
        
        frames1 = [] # stores recorded data of mic1
        frames2 = [] #stores recorded data of mic2
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
        t_start = None
        t_data_start = None
        mouse1_theta = 0
        mouse1_track_theta = 0

        def record_mouse1(): # while mouse1 is spinning
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
        
        def record_mouse2(): # while mouse2 is stopped
            while(mouse2_state != 4):
                (mouse2_state_by, mouse2_ip_address) = UDPServerSocket2.recvfrom(max_buffer_size) 
                mouse2_state_str = mouse2_state_by.decode('utf-8') #decode bytes to a string
                mouse2_state = float(mouse2_state_str)
                
                #read chunks of mic 1 and 2 simultaneously
                thread1_read2 = threading.Thread(target = read_stream1)
                thread2_read2 = threading.Thread(target = read_stream2)
                thread1_read2.start()
                thread2_read2.start()
                thread1_read2.join()
                thread2_read2.join()  

                sig1val = np.frombuffer(data1, dtype=np.int16) 
                sig1val = sig1val.astype(np.float64)
                sig1 = np.concatenate((sig1, sig1val)) #changed from append
                frames1.append(data1)

                sig2val = np.frombuffer(data2, dtype=np.int16) 
                sig2val = sig2val.astype(np.float64)
                sig2 = np.concatenate((sig2, sig2val)) #.append(sig2val) #changed from append
                frames2.append(data2)
           
                if(mouse2_state ==4):
                    print('hereeee')
                    stream1.stop_stream()
                    stream2.stop_stream()
                    stream1.close()
                    stream2.close()
                    sig1 = np.array(sig1)
                    sig2 = np.array(sig2)
                    filtered_sig1 = butter_filter(sig1,lowcut,highcut,fs,filter_order)
                    filtered_sig2 = butter_filter(sig2,lowcut,highcut,fs,filter_order)
                    
                    # state = 5 # GOT TO NEXT STATE
        thread1_record = threading.Thread(target = record_mouse1)
        thread2_record = threading.Thread(target = record_mouse2)
        thread1_record.start()
        thread2_record.start()
        thread1_record.join()
        thread2_record.join()

        state = 5 # GOT TO NEXT STATE

    elif state == 5:
        # def mouse1_next_theta():
        #     sig3_interpolater = interp1d(sig3_rms_times, sig3_rms, kind = 'nearest', bounds_error=False) #theta_list[0]=thetalist time, sig1=mic signal Try other kinds to see accuracy
        #     sig3_inp = sig3_interpolater(theta_list[0])

        def mouse2_next_theta():
            sig1 = filtered_sig1 / np.linalg.norm(filtered_sig1)  #new variable or this fine?
            sig2 = filtered_sig2 / np.linalg.norm(filtered_sig2)
        
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

        thread1_state5 = threading.Thread(target = mouse1_findtheta)
        thread2_state5 = threading.Thread(target = mouse2_next_theta)

        state = 3 # GO BACK TO STATE 3: Send packet

    else:
        print("HERE IN ELSE EXITING")
        exit()
    #once theta=2pi, exit loop -> spin is over2
    #END OF WHILE LOOP


