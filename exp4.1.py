import numpy as np
import matplotlib.pyplot as plt

# Common parameters
A0 = 1            # Amplitude
fA = 8000     # Sampling frequency (Hz)
k = np.arange(0, 61)  # Sample indices from 0 to 60
TA = 1 / fA       # Sampling period
t_k = k * TA      # Discrete time points

# --- 4.1a: f0 = 400 Hz ---
f0_a = 400
x_k_a = A0 * np.sin(2 * np.pi * f0_a * t_k)

plt.figure(figsize=(10, 4))
plt.stem(k, x_k_a, basefmt=" ")
plt.title('4.1a: Sampled Sine Wave (f0 = 400 Hz, fA = 8000 Hz)')
plt.xlabel('[k]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 4.1b: f0 = 960 Hz ---
f0_b = 960
x_k_b = A0 * np.sin(2 * np.pi * f0_b * t_k)

plt.figure(figsize=(10, 4))
plt.stem(t_k, x_k_b, basefmt=" ")
plt.title('4.1b: Sampled Sine Wave (f0 = 960 Hz, fA = 8000 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()