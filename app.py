import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dsp_utils import generate_signal, compute_fft, apply_lowpass_filter, apply_convolution, apply_window

st.title("📡 DSP Visualizer")

st.sidebar.header("Signal Settings")

signal_type = st.sidebar.selectbox(
    "Signal Type",
    ["Sine", "Square", "Chirp"]
)

freq = st.sidebar.slider("Frequency (Hz)", 1, 1000, 50)
fs = st.sidebar.slider("Sampling Frequency (Hz)", 500, 5000, 1000)
duration = st.sidebar.slider("Duration (seconds)", 1, 5, 2)

t, signal = generate_signal(signal_type, freq, fs, duration)

module = st.sidebar.selectbox(
    "Select DSP Module",
    [
        "Signal Generator",
        "FFT Spectrum",
        "Digital Filter",
        "Spectrogram",
        "Convolution",
        "Window Functions"
    ]
)

# ---------------- SIGNAL GENERATOR ----------------
if module == "Signal Generator":

    st.subheader("📈 Generated Signal")

    fig, ax = plt.subplots()
    ax.plot(t, signal)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Generated Signal")

    st.pyplot(fig)

# ---------------- FFT ----------------
elif module == "FFT Spectrum":

    st.subheader("📊 FFT Spectrum")

    freqs, magnitude = compute_fft(signal, fs)

    fig, ax = plt.subplots()
    ax.plot(freqs, magnitude)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT Spectrum")

    st.pyplot(fig)

# ---------------- FILTER ----------------
elif module == "Digital Filter":

    st.subheader("🎚 Low Pass Filter")

    cutoff = st.slider("Cutoff Frequency", 1, fs//2, 100)

    filtered = apply_lowpass_filter(signal, cutoff, fs)

    fig, ax = plt.subplots()
    ax.plot(t, signal, label="Original Signal")
    ax.plot(t, filtered, label="Filtered Signal")

    ax.legend()
    ax.set_title("Low Pass Filter Output")

    st.pyplot(fig)

# ---------------- SPECTROGRAM ----------------
elif module == "Spectrogram":

    st.subheader("🌈 Spectrogram")

    fig, ax = plt.subplots()

    ax.specgram(signal, Fs=fs)

    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Signal Spectrogram")

    st.pyplot(fig)

# ---------------- CONVOLUTION ----------------
elif module == "Convolution":

    st.subheader("🔁 Convolution")

    y = apply_convolution(signal)

    fig, ax = plt.subplots()

    ax.plot(t, signal, label="Input Signal")
    ax.plot(t, y, label="Convolved Signal")

    ax.legend()

    st.pyplot(fig)

# ---------------- WINDOW ----------------
elif module == "Window Functions":

    st.subheader("🪟 Window Functions")

    window_type = st.selectbox(
        "Select Window",
        ["hann", "hamming", "blackman"]
    )

    window = apply_window(window_type, len(signal))

    fig, ax = plt.subplots()

    ax.plot(window)

    ax.set_title(window_type + " Window")

    st.pyplot(fig)
