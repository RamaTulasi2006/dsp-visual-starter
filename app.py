import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, firwin

st.title("Interactive DSP Algorithm Visualizer")

# Sidebar controls
st.sidebar.header("Signal Settings")

signal_type = st.sidebar.selectbox(
    "Select Signal",
    ["Sine", "Cosine", "Square", "Noise"]
)

fs = st.sidebar.slider("Sampling Frequency", 100, 2000, 500)
freq = st.sidebar.slider("Signal Frequency", 1, 100, 10)
duration = st.sidebar.slider("Duration (seconds)", 1, 5, 2)

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["FFT", "DFT", "FIR Filter", "IIR Filter"]
)

cutoff = st.sidebar.slider("Cutoff Frequency", 1, 200, 50)

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


# Plot original signal
st.subheader("Time Domain Signal")

fig, ax = plt.subplots()
ax.plot(t, signal)
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
st.pyplot(fig)


# FFT
if algorithm == "FFT":

    st.subheader("Frequency Domain (FFT)")
    fft_signal = np.fft.fft(signal)
    freq_axis = np.fft.fftfreq(len(signal), 1/fs)

    fig, ax = plt.subplots()
    ax.plot(freq_axis, np.abs(fft_signal))
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    st.pyplot(fig)


# DFT
elif algorithm == "DFT":

    st.subheader("Frequency Domain (DFT)")

    N = len(signal)
    dft = []

    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += signal[n] * np.exp(-2j*np.pi*k*n/N)
        dft.append(sum_val)

    dft = np.array(dft)
    freq_axis = np.arange(N)

    fig, ax = plt.subplots()
    ax.plot(freq_axis, np.abs(dft))
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Magnitude")
    st.pyplot(fig)


# FIR Filter
elif algorithm == "FIR Filter":

    st.subheader("FIR Filter Output")

    numtaps = 51
    fir_coeff = firwin(numtaps, cutoff, fs=fs)
    filtered = lfilter(fir_coeff, 1.0, signal)

    fig, ax = plt.subplots()
    ax.plot(t, filtered)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


# IIR Filter
elif algorithm == "IIR Filter":

    st.subheader("IIR Filter Output")

    b, a = butter(4, cutoff, fs=fs)
    filtered = lfilter(b, a, signal)

    fig, ax = plt.subplots()
    ax.plot(t, filtered)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
