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

# Function to apply a rectangular filter
def apply_filter(signal, low_cutoff, high_cutoff, fs):
    # Perform Fourier Transform
    fft_signal = fft(signal)
    
    # Frequency bins
    freqs = fftfreq(len(signal), 1/fs)
    
    # Apply the filter: remove frequencies outside the range [low_cutoff, high_cutoff]
    filter_mask = (np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff)
    fft_signal_filtered = fft_signal * filter_mask
    
    # Inverse FFT to get back to time domain
    filtered_signal = ifft(fft_signal_filtered)
    return filtered_signal

# Task 1: High-pass filter to remove frequencies less than 150 Hz
cutoff_1 = 150  # Hz (high-pass filter)
filtered_signal_1 = apply_filter(signal4, cutoff_1, fs / 2, fs)

# Plot the filtered signal in time domain (Task 1)
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_1.real)
plt.title("High-pass Filtered Signal (Frequencies > 150 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Frequency analysis (Task 1: Frequency response of the filtered signal)
fft_filtered_signal_1 = fft(filtered_signal_1)
freqs = fftfreq(len(filtered_signal_1), 1/fs)
plt.figure(figsize=(10, 6))
plt.plot(freqs[:T//2], np.abs(fft_filtered_signal_1)[:T//2])  # Only positive frequencies
plt.title("Frequency Response: High-pass Filtered Signal (Frequencies > 150 Hz)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()


# Task 2: Band-pass filter to remove frequencies between 250 Hz and 350 Hz
low_cutoff_2 = 250  # Hz
high_cutoff_2 = 350  # Hz
filtered_signal_2 = apply_filter(signal4, 0, low_cutoff_2, fs)  # Low-pass filter (below 250 Hz)
filtered_signal_2 = apply_filter(filtered_signal_2, high_cutoff_2, fs / 2, fs)  # High-pass filter (above 350 Hz)

# Plot the filtered signal in time domain (Task 2)
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_2.real)
plt.title("Band-pass Filtered Signal (Frequencies < 250 Hz or > 350 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Frequency analysis (Task 2: Frequency response of the filtered signal)
fft_filtered_signal_2 = fft(filtered_signal_2)
plt.figure(figsize=(10, 6))
plt.plot(freqs[:T//2], np.abs(fft_filtered_signal_2)[:T//2])  # Only positive frequencies
plt.title("Frequency Response: Band-pass Filtered Signal (250 Hz to 350 Hz)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()


# Task 3: Low-pass filter to remove frequencies above 650 Hz
cutoff_3 = 650  # Hz (low-pass filter)
filtered_signal_3 = apply_filter(signal4, 0, cutoff_3, fs)

# Plot the filtered signal in time domain (Task 3)
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_3.real)
plt.title("Low-pass Filtered Signal (Frequencies < 650 Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Frequency analysis (Task 3: Frequency response of the filtered signal)
fft_filtered_signal_3 = fft(filtered_signal_3)
plt.figure(figsize=(10, 6))
plt.plot(freqs[:T//2], np.abs(fft_filtered_signal_3)[:T//2])  # Only positive frequencies
plt.title("Frequency Response: Low-pass Filtered Signal (Frequencies < 650 Hz)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
