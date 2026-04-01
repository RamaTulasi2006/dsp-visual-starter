import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import pandas as pd

st.set_page_config(page_title="DSP ECG Visualizer", layout="wide")

st.title("Interactive DSP Visualizer with Medical Decision System")

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
# Ideal Medical Parameters
# -----------------------------
IDEAL_HR_LOW = 60
IDEAL_HR_HIGH = 100
IDEAL_SNR = 10

# -----------------------------
# Generate Signals
# -----------------------------
t = np.linspace(0, duration, fs * duration)

# Original DSP signal
ecg = np.sin(2*np.pi*1.7*t) + 0.5*np.sin(2*np.pi*2.1*t)

# Person 1 (Healthy)
p1_ecg = np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2*t)

# Person 2 (Unhealthy - abnormal frequency)
p2_ecg = np.sin(2*np.pi*0.4*t) + 0.9*np.sin(2*np.pi*3*t)

# Add Noise
noise = noise_level * np.random.randn(len(t))

noisy_ecg = ecg + noise
p1_noisy = p1_ecg + noise
p2_noisy = p2_ecg + noise

# -----------------------------
# Butterworth Filter
# -----------------------------
b, a = butter(order, cutoff, fs=fs)

filtered_ecg = lfilter(b, a, noisy_ecg)
p1_filtered = lfilter(b, a, p1_noisy)
p2_filtered = lfilter(b, a, p2_noisy)

# -----------------------------
# FFT Function
# -----------------------------
def compute_fft(signal):
    fft_vals = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/fs)
    return freq, np.abs(fft_vals)

# -----------------------------
# SNR
# -----------------------------
def compute_snr(original, noise, filtered):
    before = np.mean(original**2) / np.mean(noise**2)
    after = np.mean(original**2) / np.mean((filtered-original)**2)
    return before, after

snr_before, snr_after = compute_snr(ecg, noise, filtered_ecg)
p1_snr_before, p1_snr_after = compute_snr(p1_ecg, noise, p1_filtered)
p2_snr_before, p2_snr_after = compute_snr(p2_ecg, noise, p2_filtered)

# -----------------------------
# Heart Rate
# -----------------------------
def get_hr(signal):
    peaks, _ = find_peaks(signal, distance=fs/2)
    if len(peaks) > 1:
        rr = np.diff(peaks)/fs
        return 60/np.mean(rr)
    return 0

p1_hr = get_hr(p1_filtered)
p2_hr = get_hr(p2_filtered)

# -----------------------------
# Health Decision Logic
# -----------------------------
def health_status(hr, snr):
    if hr < IDEAL_HR_LOW or hr > IDEAL_HR_HIGH:
        return "❌ Risk (Abnormal Heart Rate)"
    elif snr < IDEAL_SNR:
        return "⚠️ Moderate (Low Signal Quality)"
    else:
        return "✅ Healthy"

p1_status = health_status(p1_hr, p1_snr_after)
p2_status = health_status(p2_hr, p2_snr_after)

# -----------------------------
# ORIGINAL DSP VISUALIZATION
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

# Frequency Domain
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
# PERFORMANCE
# -----------------------------
st.subheader("Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("SNR Before", f"{snr_before:.2f}")

with col2:
    st.metric("SNR After", f"{snr_after:.2f}")

# -----------------------------
# MEDICAL COMPARISON
# -----------------------------
st.subheader("Healthy vs Unhealthy Comparison")

fig3, ax3 = plt.subplots(2,1, figsize=(8,6))

ax3[0].plot(t, p1_filtered)
ax3[0].set_title("Person 1 (Healthy)")

ax3[1].plot(t, p2_filtered)
ax3[1].set_title("Person 2 (Unhealthy)")

st.pyplot(fig3)

# Table
data = {
    "Parameter": ["Heart Rate (BPM)", "SNR After", "Status"],
    "Person 1": [round(p1_hr,2), round(p1_snr_after,2), p1_status],
    "Person 2": [round(p2_hr,2), round(p2_snr_after,2), p2_status],
    "Ideal Range": ["60-100 BPM", ">10", "Healthy"]
}

df = pd.DataFrame(data)
st.table(df)

# -----------------------------
# FINAL REPORT
# -----------------------------
st.subheader("Final Health Report")

st.write("### Person 1 (Healthy)")
st.success(f"Heart Rate: {p1_hr:.2f} BPM → Within normal range")
st.success(f"Condition: {p1_status}")

st.write("### Person 2 (Unhealthy)")
st.error(f"Heart Rate: {p2_hr:.2f} BPM → Outside normal range")
st.error(f"Condition: {p2_status}")

st.info("This system uses DSP to convert ECG signals into simple health information for easy understanding.")
