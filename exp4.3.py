import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 4.3.a - DFT of Sampled Sine Wave (f0 = 400 Hz, fA = 8000 Hz)
f0 = 400
fA = 8000
A0 = 1
k = np.arange(0, 61)
x = A0 * np.sin(2 * np.pi * f0 * k / fA)
X = np.fft.fft(x)
N = len(x)
freqs = np.fft.fftfreq(N, d=1/fA)

# Plot DFT of sine wave
plt.figure(figsize=(10, 4))
plt.stem(freqs[:N//2], np.abs(X[:N//2]), basefmt=" ")
plt.title("DFT of Sampled Sine Wave (f0 = 400 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4.3.b - Fourier Synthesized Sawtooth DFT Comparison

def synth_sawtooth(f0, fA, N, harmonics=4):
    t = np.arange(N) / fA
    w0 = 2 * np.pi * f0
    signal = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        coeff = (2 / np.pi) * ((-1) ** (n + 1)) / n
        signal += coeff * np.sin(n * w0 * t)
    return t, signal

# Generate signals
t1, s1 = synth_sawtooth(f0=400, fA=8000, N=400)
t2, s2 = synth_sawtooth(f0=400, fA=8000, N=16000)

# DFTs
X1 = np.fft.fft(s1)
X2 = np.fft.fft(s2)
freq1 = np.fft.fftfreq(len(s1), d=1/8000)
freq2 = np.fft.fftfreq(len(s2), d=1/8000)

# Plot 400-sample DFT
plt.figure(figsize=(10, 4))
plt.plot(freq1[:len(freq1)//2], np.abs(X1[:len(X1)//2]))
plt.title("Amplitude Spectrum (400 samples, Sawtooth)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 16000-sample DFT
plt.figure(figsize=(10, 4))
plt.plot(freq2[:len(freq2)//2], np.abs(X2[:len(X2)//2]))
plt.title("Amplitude Spectrum (16000 samples, Sawtooth)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4.3.c - DFT of Square Wave
f0 = 400
fA = 8000
N = 2000
t = np.arange(N) / fA
square_wave = signal.square(2 * np.pi * f0 * t)
X_sq = np.fft.fft(square_wave)
freqs_sq = np.fft.fftfreq(N, d=1/fA)

# Plot square wave DFT
plt.figure(figsize=(10, 4))
plt.plot(freqs_sq[:N//2], np.abs(X_sq[:N//2]))
plt.title("Amplitude Spectrum of 400 Hz Square Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
 