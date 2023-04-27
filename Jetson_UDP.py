# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:34:30 2023

@author: bakam
"""

import socket
import struct
import time

ip = '' # Bind to all network interfaces
port = 3333
max_buffer_size  = 1024

if __name__ == '__main__':
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket.bind((ip, port))

    while True:
        (message, ip_address) = UDPServerSocket.recvfrom(max_buffer_size)
        print('Message received: {}'.format(message))

        # Send two floating point numbers back
        x = [1.234, 5.678]
        print('Sending {}'.format(x))
        x = struct.pack("ff", x[0], x[1])
        UDPServerSocket.sendto(x, ip_address)