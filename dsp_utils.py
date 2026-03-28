import numpy as np
from scipy.signal import butter, lfilter

# Generate ECG Signal
def generate_ecg(fs, duration):
    t = np.linspace(0, duration, fs * duration)
    ecg = np.sin(2 * np.pi * 1.7 * t) + 0.5 * np.sin(2 * np.pi * 2.1 * t)
    return t, ecg

# Add Noise
def add_noise(signal, noise_level):
    noise = noise_level * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal, noise

# Butterworth Filter
def butter_filter(signal, fs, cutoff, order):
    b, a = butter(order, cutoff, fs=fs)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

# FFT
def compute_fft(signal, fs):
    fft_vals = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/fs)
    return freq, np.abs(fft_vals)

# SNR Calculation
def compute_snr(original, noise, filtered):
    snr_before = np.mean(original**2) / np.mean(noise**2)
    snr_after = np.mean(original**2) / np.mean((filtered - original)**2)
    return snr_before, snr_after
