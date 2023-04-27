# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:23:07 2023

@author: bakam
"""

import numpy as np
import matplotlib.pyplot as plt

# Make some sinusoids to find the timeshift between
dt = 0.001 #sampling interval
T = 2.0 # Period of the signal
t_shift_gt = -0.15 # hard coded value. We will not have this value for the signals.
f = 1 # We have different frequencies

t = np.arange(0, T, dt)
x1 = np.cos(f*2*np.pi*t)#input live recording from microphone 1
x1 = x1 / np.linalg.norm(x1) # normalize the signals

t2 = t + t_shift_gt
x2 = np.cos(f*2*np.pi*t2) # input live recording from microphone 2
x2 = x2 / np.linalg.norm(x2)

# Calculate the correlation and the corresponding timeshift corresponding to each index
C_x1x2 = np.correlate(x1, x2, mode='full')#convolving two signals without flipping
t_shift_C = np.arange(-T+dt, T, dt) # keeping track of the x axis...when you convolve the x axis changes in size

# Attempt to estimate the time shift using the correlation directly
# without normalizing by how much x1 and x2 overlapped when calculating
# each element of the correlation
i_max_C = np.argmax(C_x1x2) # What is the index of the peak?
t_shift_hat = t_shift_C[i_max_C]

# Without the per sample normalization there will be an error of
# several samples in the estimated timeshift.
# This is because the maximum of the correlation only corresponds
# to the maximum match if the signals correlated have equal magnitude
# at each shift
error = t_shift_hat - t_shift_gt
print('Estimated time shift without per shift normalization')
print('gt time shift {:0.3f} est time shift {:0.3f} error {:0.4f} s {} samples'
      .format(t_shift_gt, t_shift_hat, error, int(np.round(error/dt))))

# Calculate the magnitude of the portion of x1 that overlapped with x2
# and vice versa for each sample in C_x1x2
C_normalization_x1 = np.zeros(C_x1x2.shape[0])
C_normalization_x2 = np.zeros(C_x1x2.shape[0])

center_index = int((C_x1x2.shape[0] + 1) / 2) - 1 # Index corresponding to zero shift
low_shift_index  = -int((C_x1x2.shape[0] + 1) / 2) + 1
high_shift_index =  int((C_x1x2.shape[0] + 1) / 2) - 1
for i in range(low_shift_index, high_shift_index + 1):
    low_norm_index  = max(0,           i)
    high_norm_index = min(x1.shape[0], i + x1.shape[0])
    C_normalization_x1[i+center_index] = np.linalg.norm(x1[low_norm_index:high_norm_index])

    low_norm_index  = max(0,           -i)
    high_norm_index = min(x2.shape[0], -i + x2.shape[0])
    C_normalization_x2[i+center_index] = np.linalg.norm(x2[low_norm_index:high_norm_index])

# Normalize the calculated correlation per shift
C_x1x2_normalized_per_shift = C_x1x2 / (C_normalization_x1 * C_normalization_x2)

# Search for the maximum at most a half period back and forward due to periodicity of the input signal
# and the fact that per shift normalization causes peaks a full period to have approximately equal magnitude
# to the closest peak in the normalized correlation
max_indices_back    = -int(((1 / f) / 2) / dt) + center_index
max_indices_forward =  int(((1 / f) / 2) / dt) + center_index
i_max_C_normalized = np.argmax(C_x1x2_normalized_per_shift[max_indices_back:max_indices_forward + 1]) + max_indices_back
t_shift_hat_normalized = t_shift_C[i_max_C_normalized]

error_normalized = t_shift_hat_normalized - t_shift_gt
print('Estimated time shift with per shift normalization')
print('gt time shift {:0.3f} est time shift {:0.3f} error {:0.4f} s {} samples'
      .format(t_shift_gt, t_shift_hat_normalized, error_normalized, int(np.round(error_normalized/dt))))

plt.subplot(3,1,1)
plt.plot(t, x1)
plt.plot(t, x2)
plt.grid()
plt.title('Signals to find the time shift between')

plt.subplot(323)
plt.plot(t_shift_C, np.abs(C_x1x2))
plt.xlim([-10*np.abs(t_shift_gt), 10*np.abs(t_shift_gt)])
plt.ylim([0, 1.1])
plt.axvline(x = t_shift_gt, ymin=0, ymax=0.5, color = 'r')
plt.axvline(x = t_shift_hat, ymin=0.5, ymax=1, color = 'g')
plt.legend(['Correlation', 'gt shift', 'estimated shift'])
plt.grid()
plt.title('Correlation')

plt.subplot(325)
plt.plot(t, x1)
plt.plot(t+t_shift_hat, x2)
plt.grid()
plt.title('Aligned signals')

plt.subplot(324)
plt.plot(t_shift_C, np.abs(C_x1x2_normalized_per_shift))
plt.xlim([-10*np.abs(t_shift_gt), 10*np.abs(t_shift_gt)])
plt.ylim([0, 1.1])
plt.axvline(x = t_shift_gt, ymin=0, ymax=0.5, color = 'r')
plt.axvline(x = t_shift_hat_normalized, ymin=0.5, ymax=1, color = 'g')
plt.legend(['Correlation', 'gt shift', 'estimated shift'], loc='upper right')
plt.grid()
plt.title('Correlation Normalized Per Shift')

plt.subplot(326)
plt.plot(t, x1)
plt.plot(t+t_shift_hat_normalized, x2)
plt.grid()
plt.title('Aligned signals with per shift correlation normalization')

plt.tight_layout()
plt.show()