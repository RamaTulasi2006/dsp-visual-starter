import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, lfilter, firwin
import soundfile as sf

st.set_page_config(page_title="DSP Visualizer", layout="wide")

st.title("Interactive DSP Algorithm Visualizer")

st.sidebar.header("Signal Controls")

signal_type = st.sidebar.selectbox(
    "Signal Source",
    ["Sine", "Cosine", "Square", "Noise", "Speech-like", "ECG", "Upload Audio"]
)

fs = st.sidebar.slider("Sampling Frequency", 200, 5000, 1000)
freq = st.sidebar.slider("Signal Frequency", 1, 500, 10)
duration = st.sidebar.slider("Duration", 1, 5, 2)

algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["FFT", "DFT", "DIT FFT", "DIF FFT"]
)

filter_family = st.sidebar.selectbox(
    "Filter Family",
    ["None", "FIR", "Butterworth", "Chebyshev"]
)

filter_type = st.sidebar.selectbox(
    "Filter Type",
    ["Low Pass", "High Pass", "Band Pass", "Band Reject"]
)

cutoff = st.sidebar.slider("Cutoff Frequency", 1, 500, 50)
order = st.sidebar.slider("Filter Order", 1, 10, 4)

# Time axis
t = np.linspace(0, duration, fs*duration)

# Signal generation
if signal_type == "Sine":
    signal = np.sin(2*np.pi*freq*t)

elif signal_type == "Cosine":
    signal = np.cos(2*np.pi*freq*t)

elif signal_type == "Square":
    signal = np.sign(np.sin(2*np.pi*freq*t))

elif signal_type == "Noise":
    signal = np.random.randn(len(t))

elif signal_type == "Speech-like":
    signal = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*2*freq*t)

elif signal_type == "ECG":
    signal = np.sin(2*np.pi*1.7*t) + 0.5*np.sin(2*np.pi*2.1*t)

elif signal_type == "Upload Audio":

    uploaded = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded is not None:
        signal, fs = sf.read(uploaded)
        t = np.arange(len(signal)) / fs
    else:
        st.stop()

# Plot time-domain signal
st.subheader("Time Domain Signal")

fig1, ax1 = plt.subplots()
ax1.plot(t, signal)
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
st.pyplot(fig1)

filtered_signal = signal

# FIR filter
if filter_family == "FIR":

    coeff = firwin(51, cutoff, fs=fs)
    filtered_signal = lfilter(coeff, 1.0, signal)

# Butterworth filters
elif filter_family == "Butterworth":

    if filter_type == "Low Pass":
        b,a = butter(order, cutoff, fs=fs)

    elif filter_type == "High Pass":
        b,a = butter(order, cutoff, btype='high', fs=fs)

    elif filter_type == "Band Pass":
        b,a = butter(order, [cutoff, cutoff*2], btype='bandpass', fs=fs)

    elif filter_type == "Band Reject":
        b,a = butter(order, [cutoff, cutoff*2], btype='bandstop', fs=fs)

    filtered_signal = lfilter(b,a,signal)

# Chebyshev filters
elif filter_family == "Chebyshev":

    if filter_type == "Low Pass":
        b,a = cheby1(order,1,cutoff,fs=fs)

    elif filter_type == "High Pass":
        b,a = cheby1(order,1,cutoff,btype='high',fs=fs)

    elif filter_type == "Band Pass":
        b,a = cheby1(order,1,[cutoff,cutoff*2],btype='bandpass',fs=fs)

    elif filter_type == "Band Reject":
        b,a = cheby1(order,1,[cutoff,cutoff*2],btype='bandstop',fs=fs)

    filtered_signal = lfilter(b,a,signal)

# Plot filtered signal
st.subheader("Filtered Signal")

fig2, ax2 = plt.subplots()
ax2.plot(t, filtered_signal)
ax2.set_xlabel("Time")
ax2.set_ylabel("Amplitude")
st.pyplot(fig2)

# Frequency domain
st.subheader("Frequency Spectrum")

spectrum = np.fft.fft(filtered_signal)
freq_axis = np.fft.fftfreq(len(filtered_signal),1/fs)

fig3, ax3 = plt.subplots()
ax3.plot(freq_axis, np.abs(spectrum))
ax3.set_xlabel("Frequency")
ax3.set_ylabel("Magnitude")
st.pyplot(fig3)
