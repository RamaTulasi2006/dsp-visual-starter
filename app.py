import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import pandas as pd

st.set_page_config(page_title="ECG Health Analyzer", layout="wide")

st.title("ECG Health Analysis using DSP (Butterworth Filter)")

# Sidebar Controls
st.sidebar.header("Signal Parameters")

fs = st.sidebar.slider("Sampling Frequency (Hz)", 500, 5000, 1000)
duration = st.sidebar.slider("Duration (seconds)", 2, 5, 3)
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5)

st.sidebar.header("Filter Parameters")

cutoff = st.sidebar.slider("Cutoff Frequency (Hz)", 1, 100, 40)
order = st.sidebar.slider("Filter Order", 1, 10, 4)

# Time axis
t = np.linspace(0, duration, fs * duration)

# -----------------------------
# Healthy ECG Signal
# -----------------------------
healthy_ecg = np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2*t)

# -----------------------------
# Unhealthy ECG Signal
# -----------------------------
unhealthy_ecg = np.sin(2*np.pi*0.8*t) + 0.8*np.sin(2*np.pi*3*t)

# Add Noise
noise = noise_level * np.random.randn(len(t))

healthy_noisy = healthy_ecg + noise
unhealthy_noisy = unhealthy_ecg + noise

# Butterworth Filter
b, a = butter(order, cutoff, fs=fs)

healthy_filtered = lfilter(b, a, healthy_noisy)
unhealthy_filtered = lfilter(b, a, unhealthy_noisy)

# FFT Function
def compute_fft(signal):
    fft_vals = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/fs)
    return freq, np.abs(fft_vals)

# SNR
def compute_snr(original, noise, filtered):
    before = np.mean(original**2) / np.mean(noise**2)
    after = np.mean(original**2) / np.mean((filtered - original)**2)
    return before, after

h_snr_before, h_snr_after = compute_snr(healthy_ecg, noise, healthy_filtered)
u_snr_before, u_snr_after = compute_snr(unhealthy_ecg, noise, unhealthy_filtered)

# Heart Rate Calculation
def get_heart_rate(signal):
    peaks, _ = find_peaks(signal, distance=fs/2)
    if len(peaks) > 1:
        rr = np.diff(peaks)/fs
        return 60/np.mean(rr)
    return 0

h_hr = get_heart_rate(healthy_filtered)
u_hr = get_heart_rate(unhealthy_filtered)

# Signal Quality
def quality(snr):
    if snr > 10:
        return "Good"
    elif snr > 5:
        return "Moderate"
    return "Poor"

# -----------------------------
# Plot Signals
# -----------------------------
st.subheader("Healthy vs Unhealthy ECG (Time Domain)")

fig, ax = plt.subplots(2,1, figsize=(8,6))

ax[0].plot(t, healthy_filtered)
ax[0].set_title("Healthy ECG")

ax[1].plot(t, unhealthy_filtered)
ax[1].set_title("Unhealthy ECG")

st.pyplot(fig)

# -----------------------------
# Comparison Table
# -----------------------------
st.subheader("Comparison Table")

data = {
    "Parameter": ["Heart Rate (BPM)", "SNR Before", "SNR After", "Signal Quality"],
    "Healthy Person": [round(h_hr,2), round(h_snr_before,2), round(h_snr_after,2), quality(h_snr_after)],
    "Unhealthy Person": [round(u_hr,2), round(u_snr_before,2), round(u_snr_after,2), quality(u_snr_after)]
}

df = pd.DataFrame(data)
st.table(df)

# -----------------------------
# Simple Report
# -----------------------------
st.subheader("Simple Health Report")

if h_hr >= 60 and h_hr <= 100:
    st.success("Healthy Person: Normal Heart Rate")
else:
    st.warning("Healthy Person: Abnormal Heart Rate")

if u_hr < 60:
    st.error("Unhealthy Person: Low Heart Rate (Bradycardia)")
elif u_hr > 100:
    st.error("Unhealthy Person: High Heart Rate (Tachycardia)")
else:
    st.warning("Unhealthy Person: Needs further diagnosis")

# -----------------------------
# Final Message
# -----------------------------
st.info("This system helps even non-technical users understand ECG signals using simple parameters like heart rate and signal quality.")
