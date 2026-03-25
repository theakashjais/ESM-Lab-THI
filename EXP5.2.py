import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# Parameters
fs = 10000  # Sampling frequency (10 kHz)
T = 2000  # Number of samples
t = np.arange(T) / fs  # Time vector

# Signals
signal1 = 50 * np.cos(2 * np.pi * 100 * t)  # 100 Hz cosine
signal2 = 100 * np.sin(2 * np.pi * 300 * t)  # 300 Hz sine
signal3 = 200 * np.sin(2 * np.pi * 700 * t)  # 700 Hz sine
signal4 = signal1 + signal2 + signal3  # Sum of all signals

# Function to apply rectangular filter
def rectangular_filter(signal, low_cutoff, high_cutoff, fs):
    # Fourier transform of the signal
    fft_signal = fft(signal)
    
    # Frequency bins
    freqs = fftfreq(len(signal), 1/fs)
    
    # Apply the filter: remove frequencies outside the range [low_cutoff, high_cutoff]
    filter_mask = (np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff)
    
    # Apply filter to the frequency domain
    fft_signal_filtered = fft_signal * filter_mask
    
    # Inverse FFT to get back to time domain
    filtered_signal = ifft(fft_signal_filtered)
    return filtered_signal

# Task 1: Filter out all oscillations less than 150 Hz
low_cutoff_1 = 150  # Cutoff for low frequencies
filtered_signal_1 = rectangular_filter(signal4, low_cutoff_1, fs / 2, fs)

# Plot filtered signal in time domain
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_1.real)
plt.title("Filtered Signal (Frequencies > 150 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Task 2: Filter out frequencies between 250 Hz to 350 Hz
filtered_signal_2 = rectangular_filter(signal4, 0, 250, fs)  # Low-pass filter (below 250 Hz)
filtered_signal_2 = rectangular_filter(filtered_signal_2, 350, fs / 2, fs)  # High-pass filter (above 350 Hz)

# Plot filtered signal in time domain
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_2.real)
plt.title("Filtered Signal (Frequencies < 250 Hz or > 350 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Task 3: Filter out frequencies higher than 650 Hz
filtered_signal_3 = rectangular_filter(signal4, 0, 650, fs)  # Low-pass filter (below 650 Hz)

# Plot filtered signal in time domain
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_3.real)
plt.title("Filtered Signal (Frequencies < 650 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
