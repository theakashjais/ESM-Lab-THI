import numpy as np
import matplotlib.pyplot as plt

# Parameters for original signal
f_signal = 960  # Hz
amplitude = 2
num_periods = 5
fa = 100000  # Hz
T_signal = 1 / f_signal
Ta = 1 / fa
samples_per_period = int(T_signal / Ta)
total_samples = samples_per_period * num_periods

# a) Create sawtooth signal
one_period = np.linspace(0, 1, samples_per_period, endpoint=False)
signal = np.tile(one_period, num_periods)
signal *= amplitude  # Scale to desired amplitude

t = np.linspace(0, num_periods * T_signal, total_samples, endpoint=False)

# b) Plot original sawtooth signal
plt.figure(figsize=(10, 6))
plt.plot(t, signal)
plt.title('Sawtooth Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# c) Fourier Synthesis - New time vector independent of original
f0 = 400  # Fundamental frequency for synthesis
T = 1 / f0
t_synth = np.linspace(0, 5 * T, 1000)
ideal_sawtooth = 2 * (t_synth / T - np.floor(0.5 + t_synth / T))

plt.figure(figsize=(12, 6))
plt.plot(t_synth, ideal_sawtooth, label='Original Sawtooth', linewidth=1)

# Add approximations with harmonics
for n in range(1, 6):
    approx = np.zeros_like(t_synth)
    for k in range(1, n + 1):
        approx += (-1)**(k+1) * (2 / (k * np.pi)) * np.sin(2 * np.pi * k * f0 * t_synth)
    plt.plot(t_synth, approx, label=f'{n} Harmonics')

plt.title('Sawtooth Synthesis with Fourier Series')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# d) Time shift the sawtooth signal
time_shift = 0.0015  # seconds
shift_samples = int(time_shift / Ta)
shifted_signal = np.roll(signal, shift_samples)

plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Original Sawtooth')
plt.plot(t, shifted_signal, label=f'Shifted Sawtooth ({time_shift} s)')
plt.title('Sawtooth Signal and Time-Shifted Version')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()