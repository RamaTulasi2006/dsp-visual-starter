import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import pandas as pd

st.set_page_config(page_title="ECG Health Analyzer", layout="wide")

st.title("Interactive DSP Visualizer with Medical Decision System")

# -----------------------------
# Sidebar
# -----------------------------
fs = st.sidebar.slider("Sampling Frequency", 500, 5000, 1000)
duration = st.sidebar.slider("Duration", 2, 5, 3)
noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5)

cutoff = st.sidebar.slider("Cutoff Frequency", 1, 100, 40)
order = st.sidebar.slider("Filter Order", 1, 10, 4)

# Ideal Parameters
IDEAL_HR_LOW = 60
IDEAL_HR_HIGH = 100
IDEAL_SNR = 10

# -----------------------------
# Signals
# -----------------------------
t = np.linspace(0, duration, fs * duration)

# Person 1 (Healthy)
p1_ecg = np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2*t)

# Person 2 (Unhealthy)
p2_ecg = np.sin(2*np.pi*0.6*t) + 0.9*np.sin(2*np.pi*3*t)

# Noise
noise = noise_level * np.random.randn(len(t))

p1_noisy = p1_ecg + noise
p2_noisy = p2_ecg + noise

# Filter
b, a = butter(order, cutoff, fs=fs)
p1_filtered = lfilter(b, a, p1_noisy)
p2_filtered = lfilter(b, a, p2_noisy)

# -----------------------------
# Functions
# -----------------------------
def get_hr(signal):
    peaks, _ = find_peaks(signal, distance=fs/2)
    if len(peaks) > 1:
        rr = np.diff(peaks)/fs
        return 60/np.mean(rr)
    return 0

def compute_snr(original, noise, filtered):
    return np.mean(original**2) / np.mean((filtered-original)**2)

def health_status(hr, snr):
    if hr < IDEAL_HR_LOW or hr > IDEAL_HR_HIGH:
        return "❌ Risk (Abnormal Heart Rate)"
    elif snr < IDEAL_SNR:
        return "⚠️ Moderate (Low Signal Quality)"
    else:
        return "✅ Healthy"

# -----------------------------
# Parameters
# -----------------------------
p1_hr = get_hr(p1_filtered)
p2_hr = get_hr(p2_filtered)

p1_snr = compute_snr(p1_ecg, noise, p1_filtered)
p2_snr = compute_snr(p2_ecg, noise, p2_filtered)

p1_status = health_status(p1_hr, p1_snr)
p2_status = health_status(p2_hr, p2_snr)

# -----------------------------
# Visualization
# -----------------------------
st.subheader("ECG Signals")

fig, ax = plt.subplots(2,1, figsize=(8,6))

ax[0].plot(t, p1_filtered)
ax[0].set_title("Person 1 ECG")

ax[1].plot(t, p2_filtered)
ax[1].set_title("Person 2 ECG")

st.pyplot(fig)

# -----------------------------
# Comparison Table
# -----------------------------
st.subheader("Medical Comparison Table")

data = {
    "Parameter": ["Heart Rate (BPM)", "SNR", "Status"],
    "Person 1": [round(p1_hr,2), round(p1_snr,2), p1_status],
    "Person 2": [round(p2_hr,2), round(p2_snr,2), p2_status],
    "Ideal Range": ["60-100 BPM", ">10", "Healthy"]
}

df = pd.DataFrame(data)
st.table(df)

# -----------------------------
# Final Report
# -----------------------------
st.subheader("Final Health Report")

st.write("### Person 1:")
st.write(f"Heart Rate: {p1_hr:.2f} BPM")
st.write(f"Signal Quality (SNR): {p1_snr:.2f}")
st.write(f"Condition: {p1_status}")

st.write("### Person 2:")
st.write(f"Heart Rate: {p2_hr:.2f} BPM")
st.write(f"Signal Quality (SNR): {p2_snr:.2f}")
st.write(f"Condition: {p2_status}")

# -----------------------------
# Easy Explanation
# -----------------------------
st.info("This system compares ECG signals with ideal medical values and gives a simple health report understandable by anyone.")
