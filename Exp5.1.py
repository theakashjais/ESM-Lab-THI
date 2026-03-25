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

# Plot the time-domain signals
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(t, signal1)
plt.title("Signal 1: 100 Hz Cosine")
plt.subplot(4, 1, 2)
plt.plot(t, signal2)
plt.title("Signal 2: 300 Hz Sine")
plt.subplot(4, 1, 3)
plt.plot(t, signal3)
plt.title("Signal 3: 700 Hz Sine")
plt.subplot(4, 1, 4)
plt.plot(t, signal4)
plt.title("Signal 4: Sum of Signals 1, 2, and 3")
plt.tight_layout()
plt.show()

# Apply Fourier Transform to each signal
fft_signal1 = fft(signal1)
fft_signal2 = fft(signal2)
fft_signal3 = fft(signal3)
fft_signal4 = fft(signal4)

# Frequency axis
freqs = fftfreq(T, 1/fs)

# Plot frequency-domain (FFT) results for each signal
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(freqs[:T//2], np.abs(fft_signal1)[:T//2])  # Plot only positive frequencies
plt.title("FFT of Signal 1: 100 Hz Cosine")
plt.subplot(4, 1, 2)
plt.plot(freqs[:T//2], np.abs(fft_signal2)[:T//2])
plt.title("FFT of Signal 2: 300 Hz Sine")
plt.subplot(4, 1, 3)
plt.plot(freqs[:T//2], np.abs(fft_signal3)[:T//2])
plt.title("FFT of Signal 3: 700 Hz Sine")
plt.subplot(4, 1, 4)
plt.plot(freqs[:T//2], np.abs(fft_signal4)[:T//2])
plt.title("FFT of Signal 4: Sum of Signals 1, 2, and 3")
plt.tight_layout()
plt.show()

# Sum the Fourier transforms and apply inverse Fourier transform
fft_sum = fft_signal1 + fft_signal2 + fft_signal3
reconstructed_signal = ifft(fft_sum)

# Plot the reconstructed signal and compare with signal 4
plt.figure(figsize=(10, 6))
plt.plot(t, reconstructed_signal.real, label='Reconstructed Signal')
plt.plot(t, signal4, '--', label='Original Sum of Signals')
plt.title("Reconstructed Signal vs. Original Signal")
plt.legend()
plt.grid(True)
plt.show()
