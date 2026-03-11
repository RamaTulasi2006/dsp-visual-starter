import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

st.title("Communication Signal Noise Reduction using DSP")

st.sidebar.header("Signal Parameters")

# Parameters
fs = st.sidebar.slider("Sampling Frequency (Hz)", 500, 5000, 1000)
f = st.sidebar.slider("Signal Frequency (Hz)", 1, 200, 50)
duration = st.sidebar.slider("Signal Duration (seconds)", 1, 5, 2)
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5)

st.sidebar.header("Filter Parameters")

cutoff = st.sidebar.slider("Cutoff Frequency", 1, 200, 100)
order = st.sidebar.slider("Filter Order", 1, 10, 4)

# Time axis
t = np.linspace(0, duration, fs*duration)

# Original Signal
signal = np.sin(2*np.pi*f*t)

# Add Noise
noise = noise_level * np.random.randn(len(t))
noisy_signal = signal + noise

# Butterworth Filter
b, a = butter(order, cutoff, fs=fs)
filtered_signal = lfilter(b, a, noisy_signal)

# FFT
def compute_fft(sig):
    spectrum = np.fft.fft(sig)
    freq_axis = np.fft.fftfreq(len(sig), 1/fs)
    return freq_axis, np.abs(spectrum)

freq1, spec1 = compute_fft(signal)
freq2, spec2 = compute_fft(noisy_signal)
freq3, spec3 = compute_fft(filtered_signal)

# Plot signals
st.subheader("Time Domain Comparison")

fig, ax = plt.subplots(3,1, figsize=(8,8))

ax[0].plot(t, signal)
ax[0].set_title("Original Signal")

ax[1].plot(t, noisy_signal)
ax[1].set_title("Noisy Signal (Communication Channel)")

ax[2].plot(t, filtered_signal)
ax[2].set_title("Filtered Signal (After DSP)")

st.pyplot(fig)

# Frequency spectrum
st.subheader("Frequency Domain Comparison")

fig2, ax2 = plt.subplots(3,1, figsize=(8,8))

ax2[0].plot(freq1, spec1)
ax2[0].set_title("Original Spectrum")

ax2[1].plot(freq2, spec2)
ax2[1].set_title("Noisy Spectrum")

ax2[2].plot(freq3, spec3)
ax2[2].set_title("Filtered Spectrum")

st.pyplot(fig2)

# Comparison metrics
snr_before = np.mean(signal**2) / np.mean(noise**2)
snr_after = np.mean(signal**2) / np.mean((filtered_signal-signal)**2)

st.subheader("Performance Comparison")

st.write("SNR Before Filtering:", round(snr_before,2))
st.write("SNR After Filtering:", round(snr_after,2))
