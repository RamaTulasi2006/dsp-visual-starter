import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import platform

def available_signals():
    return ["sine", "square", "triangle", "sawtooth", "chirp"]

def available_windows():
    return ["boxcar", "hann", "hamming", "blackman"]

def generate_signal(sig_type, fs, duration, amp=1.0, f=100, phase_deg=0, duty=50, f_start=None, f_end=None):
    t = np.arange(0, duration, 1/fs)
    phase = np.deg2rad(phase_deg)

    if sig_type == "sine":
        x = amp * np.sin(2*np.pi*f*t + phase)
    elif sig_type == "square":
        x = amp * signal.square(2*np.pi*f*t + phase, duty=duty/100)
    elif sig_type == "triangle":
        x = amp * signal.sawtooth(2*np.pi*f*t + phase, width=0.5)
    elif sig_type == "sawtooth":
        x = amp * signal.sawtooth(2*np.pi*f*t + phase)
    elif sig_type == "chirp":
        if f_start is None: f_start = 10
        if f_end is None: f_end = 1000
        x = amp * signal.chirp(t, f_start, duration, f_end)
    else:
        x = np.zeros_like(t)

    return t, x

def fft_analyze(x, fs, nfft=2048, window_name="hann"):
    win = signal.get_window(window_name, len(x))
    xw = x * win
    X = np.fft.rfft(xw, nfft)
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    return freqs, X

def design_fir(band, fs, order, f1, f2=None):
    if band == "lowpass":
        b = signal.firwin(order, f1, fs=fs)
    elif band == "highpass":
        b = signal.firwin(order, f1, pass_zero=False, fs=fs)
    elif band == "bandpass":
        b = signal.firwin(order, [f1, f2], pass_zero=False, fs=fs)
    elif band == "bandstop":
        b = signal.firwin(order, [f1, f2], pass_zero=True, fs=fs)
    return b

def design_iir(band, fs, order, f1, f2=None):
    if band == "lowpass":
        b, a = signal.butter(order, f1, btype="lowpass", fs=fs)
    elif band == "highpass":
        b, a = signal.butter(order, f1, btype="highpass", fs=fs)
    elif band == "bandpass":
        b, a = signal.butter(order, [f1, f2], btype="bandpass", fs=fs)
    elif band == "bandstop":
        b, a = signal.butter(order, [f1, f2], btype="bandstop", fs=fs)
    return b, a

def apply_filter(b, a, x):
    return signal.lfilter(b, a, x)

def freqz_response(b, a, worN=2048, fs=1):
    return signal.freqz(b, a, worN=worN, fs=fs)

def convolve_signals(x, h):
    return np.convolve(x, h, mode="full")

def plot_spectrogram(x, fs):
    fig, ax = plt.subplots(figsize=(7,3))
    ax.specgram(x, NFFT=256, Fs=fs, noverlap=128)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    return fig

def version_report():
    return f"""
Python : {platform.python_version()}
NumPy  : {np.__version__}
Platform : {platform.system()}
"""
