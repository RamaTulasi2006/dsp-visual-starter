import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import pandas as pd

st.set_page_config(page_title="DSP ECG Visualizer", layout="wide")

st.title("Interactive DSP Algorithm Visualizer with Medical Analysis")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Signal Parameters")

fs = st.sidebar.slider("Sampling Frequency (Hz)", 500, 5000, 1000)
duration = st.sidebar.slider("Duration (seconds)", 2, 5, 3)
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5)

st.sidebar.header("Filter Parameters")

cutoff = st.sidebar.slider("Cutoff Frequency (Hz)", 1, 100, 40)
order = st.sidebar.slider("Filter Order", 1, 10, 4)

# -----------------------------
# Generate Signals
# -----------------------------
t = np.linspace(0, duration, fs * duration)

# Original ECG (base)
ecg = np.sin(2*np.pi*1.7*t) + 0.5*np.sin(2*np.pi*2.1*t)

# Healthy & Unhealthy
healthy_ecg = np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2*t)
unhealthy_ecg = np.sin(2*np.pi*0.8*t) + 0.8*np.sin(2*np.pi*3*t)

# Add Noise
noise = noise_level * np.random.randn(len(t))
noisy_ecg = ecg + noise
healthy_noisy = healthy_ecg + noise
unhealthy_noisy = unhealthy_ecg + noise

# -----------------------------
# Butterworth Filter
# -----------------------------
b, a = butter(order, cutoff, fs=fs)

filtered_ecg = lfilter(b, a, noisy_ecg)
healthy_filtered = lfilter(b, a, healthy_noisy)
unhealthy_filtered = lfilter(b, a, unhealthy_noisy)

# -----------------------------
# FFT Function
# -----------------------------
def compute_fft(signal):
    fft_vals = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/fs)
    return freq, np.abs(fft_vals)

# -----------------------------
# SNR Calculation
# -----------------------------
def compute_snr(original, noise, filtered):
    before = np.mean(original**2) / np.mean(noise**2)
    after = np.mean(original**2) / np.mean((filtered - original)**2)
    return before, after

snr_before, snr_after = compute_snr(ecg, noise, filtered_ecg)
h_snr_before, h_snr_after = compute_snr(healthy_ecg, noise, healthy_filtered)
u_snr_before, u_snr_after = compute_snr(unhealthy_ecg, noise, unhealthy_filtered)

# -----------------------------
# Heart Rate Calculation
# -----------------------------
def get_hr(signal):
    peaks, _ = find_peaks(signal, distance=fs/2)
    if len(peaks) > 1:
        rr = np.diff(peaks)/fs
        return 60/np.mean(rr)
    return 0

h_hr = get_hr(healthy_filtered)
u_hr = get_hr(unhealthy_filtered)

# -----------------------------
# Signal Quality
# -----------------------------
def quality(snr):
    if snr > 10:
        return "Good"
    elif snr > 5:
        return "Moderate"
    return "Poor"

# -----------------------------
# OLD VISUALIZATION (UNCHANGED)
# -----------------------------
st.subheader("Original DSP Visualization")

fig1, ax1 = plt.subplots(3,1, figsize=(8,8))

ax1[0].plot(t, ecg)
ax1[0].set_title("Original Signal")

ax1[1].plot(t, noisy_ecg)
ax1[1].set_title("Noisy Signal")

ax1[2].plot(t, filtered_ecg)
ax1[2].set_title("Filtered Signal")

st.pyplot(fig1)

# Frequency
f1, s1 = compute_fft(ecg)
f2, s2 = compute_fft(noisy_ecg)
f3, s3 = compute_fft(filtered_ecg)

st.subheader("Frequency Domain")

fig2, ax2 = plt.subplots(3,1, figsize=(8,8))

ax2[0].plot(f1, s1)
ax2[0].set_title("Original Spectrum")

ax2[1].plot(f2, s2)
ax2[1].set_title("Noisy Spectrum")

ax2[2].plot(f3, s3)
ax2[2].set_title("Filtered Spectrum")

st.pyplot(fig2)

# -----------------------------
# PERFORMANCE METRICS
# -----------------------------
st.subheader("Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("SNR Before", f"{snr_before:.2f}")

with col2:
    st.metric("SNR After", f"{snr_after:.2f}")

# -----------------------------
# NEW MEDICAL ANALYSIS
# -----------------------------
st.subheader("Medical Comparison")

fig3, ax3 = plt.subplots(2,1, figsize=(8,6))

ax3[0].plot(t, healthy_filtered)
ax3[0].set_title("Healthy ECG")

ax3[1].plot(t, unhealthy_filtered)
ax3[1].set_title("Unhealthy ECG")

st.pyplot(fig3)

# Table
data = {
    "Parameter": ["Heart Rate (BPM)", "SNR After", "Signal Quality"],
    "Healthy": [round(h_hr,2), round(h_snr_after,2), quality(h_snr_after)],
    "Unhealthy": [round(u_hr,2), round(u_snr_after,2), quality(u_snr_after)]
}

df = pd.DataFrame(data)
st.table(df)

# -----------------------------
# REPORT
# -----------------------------
st.subheader("Simple Health Report")

if 60 <= h_hr <= 100:
    st.success("Healthy Person: Normal Heart Rate")
else:
    st.warning("Healthy Person: Abnormal")

if u_hr < 60:
    st.error("Unhealthy Person: Low Heart Rate")
elif u_hr > 100:
    st.error("Unhealthy Person: High Heart Rate")
else:
    st.warning("Unhealthy Person: Needs Check")

st.info("This system converts complex ECG signals into simple medical understanding for all users.")
