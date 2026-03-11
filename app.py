import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, firwin

st.set_page_config(page_title="DSP Algorithm Visualizer", layout="wide")

st.title("Interactive DSP Algorithm Visualizer")

st.sidebar.header("Signal Controls")

# Signal selection
signal_type = st.sidebar.selectbox(
    "Select Signal",
    ["Sine", "Cosine", "Square", "Noise", "Speech-like"]
)

fs = st.sidebar.slider("Sampling Frequency", 200, 4000, 1000)
freq = st.sidebar.slider("Signal Frequency", 1, 200, 10)
duration = st.sidebar.slider("Signal Duration (seconds)", 1, 5, 2)

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["FFT", "DFT"]
)

filter_type = st.sidebar.selectbox(
    "Select Filter",
    ["None", "FIR Low Pass", "Butterworth Low Pass"]
)

cutoff = st.sidebar.slider("Cutoff Frequency", 1, 300, 50)

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

# Plot original signal
st.subheader("Original Time Domain Signal")

fig1, ax1 = plt.subplots()
ax1.plot(t, signal)
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
st.pyplot(fig1)

# Apply Filters
filtered_signal = signal

if filter_type == "FIR Low Pass":

    numtaps = 51
    fir_coeff = firwin(numtaps, cutoff, fs=fs)
    filtered_signal = lfilter(fir_coeff, 1.0, signal)

elif filter_type == "Butterworth Low Pass":

    b, a = butter(4, cutoff, fs=fs)
    filtered_signal = lfilter(b, a, signal)

# Plot filtered signal
st.subheader("Filtered Signal")

fig2, ax2 = plt.subplots()
ax2.plot(t, filtered_signal)
ax2.set_xlabel("Time")
ax2.set_ylabel("Amplitude")
st.pyplot(fig2)

# FFT
if algorithm == "FFT":

    st.subheader("Frequency Domain using FFT")

    fft_vals = np.fft.fft(filtered_signal)
    freqs = np.fft.fftfreq(len(filtered_signal), 1/fs)

    fig3, ax3 = plt.subplots()
    ax3.plot(freqs, np.abs(fft_vals))
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Magnitude")
    st.pyplot(fig3)

# DFT
elif algorithm == "DFT":

    st.subheader("Frequency Domain using DFT")

    N = len(filtered_signal)
    dft = []

    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += filtered_signal[n] * np.exp(-2j*np.pi*k*n/N)
        dft.append(sum_val)

    dft = np.array(dft)

    fig4, ax4 = plt.subplots()
    ax4.plot(np.abs(dft))
    ax4.set_xlabel("Frequency Bin")
    ax4.set_ylabel("Magnitude")
    st.pyplot(fig4)
