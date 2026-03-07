import numpy as np
from scipy import signal


# SIGNAL GENERATOR
def generate_signal(signal_type, freq, fs, duration):

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if signal_type == "Sine":
        x = np.sin(2 * np.pi * freq * t)

    elif signal_type == "Square":
        x = signal.square(2 * np.pi * freq * t)

    elif signal_type == "Chirp":
        x = signal.chirp(t, f0=1, f1=freq, t1=duration)

    else:
        x = np.sin(2 * np.pi * freq * t)

    return t, x


# FFT
def compute_fft(x, fs):

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(X), 1 / fs)

    mask = freqs >= 0

    return freqs[mask], np.abs(X[mask])


# DIGITAL FILTER
def apply_lowpass_filter(x, cutoff, fs):

    b, a = signal.butter(4, cutoff / (fs / 2), btype="low")

    y = signal.lfilter(b, a, x)

    return y


# CONVOLUTION
def apply_convolution(x):

    h = np.ones(20) / 20

    y = np.convolve(x, h, mode="same")

    return y


# WINDOW FUNCTIONS
def apply_window(window_type, length):

    window = signal.get_window(window_type, length)

    return window
