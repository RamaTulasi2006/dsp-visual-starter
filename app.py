import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Page config
st.set_page_config(page_title="ECG DSP Visualizer", layout="wide")

# Title
st.title("ECG Signal Noise Removal using Butterworth Filter")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Signal Parameters")

fs = st.sidebar.slider("Sampling Frequency (Hz)", 500, 5000, 1000)
duration = st.sidebar.slider("Duration (seconds)", 1, 5, 2)
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5)

st.sidebar.header("Filter Parameters")

cutoff = st.sidebar.slider("Cutoff Frequency (Hz)", 1, 100, 40)
order = st.sidebar.slider("Filter Order", 1, 10, 4)

# -----------------------------
# Generate ECG Signal
# -----------------------------
t = np.linspace(0, duration, fs * duration)

# Simulated ECG waveform
ecg = np.sin(2 * np.pi * 1.7 * t) + 0.5 * np.sin(2 * np.pi * 2.1 * t)

# Add noise
noise = noise_level * np.random.randn(len(t))
noisy_ecg = ecg + noise

# -----------------------------
# Apply Butterworth Filter
# -----------------------------
b, a = butter(order, cutoff, fs=fs)
filtered_ecg = lfilter(b, a, noisy_ecg)

# -----------------------------
# FFT Function
# -----------------------------
def compute_fft(signal):
    fft_vals = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/fs)
    return freq, np.abs(fft_vals)

f1, s1 = compute_fft(ecg)
f2, s2 = compute_fft(noisy_ecg)
f3, s3 = compute_fft(filtered_ecg)

# -----------------------------
# Time Domain Plots
# -----------------------------
st.subheader("Time Domain Comparison")

fig1, ax1 = plt.subplots(3, 1, figsize=(8, 8))

ax1[0].plot(t, ecg)
ax1[0].set_title("Original ECG Signal")

ax1[1].plot(t, noisy_ecg)
ax1[1].set_title("Noisy ECG Signal")

ax1[2].plot(t, filtered_ecg)
ax1[2].set_title("Filtered ECG Signal")

plt.tight_layout()
st.pyplot(fig1)

# -----------------------------
# Frequency Domain Plots
# -----------------------------
st.subheader("Frequency Domain Comparison")

fig2, ax2 = plt.subplots(3, 1, figsize=(8, 8))

ax2[0].plot(f1, s1)
ax2[0].set_title("Original Spectrum")

ax2[1].plot(f2, s2)
ax2[1].set_title("Noisy Spectrum")

ax2[2].plot(f3, s3)
ax2[2].set_title("Filtered Spectrum")

plt.tight_layout()
st.pyplot(fig2)

# -----------------------------
# SNR Calculation
# -----------------------------
snr_before = np.mean(ecg**2) / np.mean(noise**2)
snr_after = np.mean(ecg**2) / np.mean((filtered_ecg - ecg)**2)

# -----------------------------
# Metrics Display
# -----------------------------
st.subheader("Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.metric("SNR Before Filtering", f"{snr_before:.2f}")

with col2:
    st.metric("SNR After Filtering", f"{snr_after:.2f}")
