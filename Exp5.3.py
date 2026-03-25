import numpy as np
import matplotlib.pyplot as plt

# Sampling rate and time vector
Fs = 10000  # Sampling frequency (10 kHz)
T = 2000    # Number of samples
t = np.arange(T) / Fs  # Time vector

# Creating the signal with frequencies from 10 Hz to 1000 Hz
f1, f2, f3 = 100, 300, 700  # Frequencies
A1, A2, A3 = 50, 100, 200   # Amplitudes

signal = A1 * np.cos(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + A3 * np.sin(2 * np.pi * f3 * t)

# Fourier Transform of the signal
signal_fft = np.fft.fft(signal)

# Frequency vector
freq = np.fft.fftfreq(len(signal), 1/Fs)

# Apply the Low-Pass Filter (0-100 Hz)
lpf = np.abs(freq) < 100
filtered_signal_lpf = signal_fft * lpf

# Apply the Band-Pass Filter (400-450 Hz)
bpf = (np.abs(freq) > 400) & (np.abs(freq) < 450)
filtered_signal_bpf = signal_fft * bpf

# Combine the two filters
combined_filtered_signal = filtered_signal_lpf + filtered_signal_bpf

# Inverse FFT to get the time domain signal
time_signal = np.fft.ifft(combined_filtered_signal)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label="Original Signal")
plt.title("Original Signal in Time Domain")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, time_signal.real, label="Filtered Signal")
plt.title("Filtered Signal in Time Domain")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
