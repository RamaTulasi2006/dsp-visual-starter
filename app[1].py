import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal

from dsp_utils import (
    generate_signal, available_signals, available_windows,
    fft_analyze, design_fir, design_iir, apply_filter,
    plot_spectrogram, freqz_response, convolve_signals,
    version_report,
)

st.set_page_config(page_title="DSP Visualizer", layout="wide")

st.title("📈 Interactive DSP Algorithm Visualizer")
st.caption("Explore signals, sampling, FFT/windowing, filters, convolution, and spectrograms.")

with st.sidebar:
    st.header("⚙️ Global Settings")
    duration = st.slider("Signal duration (s)", 0.01, 2.0, 0.5, 0.01)
    fs = st.slider("Sampling rate (Hz)", 200, 50000, 5000, 100)
    nfft = st.select_slider("FFT size (N)", options=[256, 512, 1024, 2048, 4096, 8192], value=2048)
    noise_power = st.slider("Additive white noise power (0=noise-free)", 0.0, 0.1, 0.0, 0.001)
    window_name = st.selectbox("Window for FFT", available_windows(), index=1)
    st.markdown("---")
    st.subheader("ℹ️ Environment")
    st.code(version_report(), language="text")

sig_tab, alias_tab, fft_tab, filt_tab, conv_tab, spec_tab = st.tabs([
    "Signal", "Sampling/Aliasing", "FFT & Windowing", "Filtering", "Convolution", "Spectrogram"
])

with sig_tab:
    st.subheader("Signal Generator")
    col1, col2 = st.columns(2)
    with col1:
        sig_type = st.selectbox("Signal type", available_signals())
        freq = st.slider("Frequency f (Hz)", 0.0, fs/2, min(100.0, fs/2), 1.0)
        amp = st.slider("Amplitude", 0.0, 5.0, 1.0, 0.1)
        phase = st.slider("Phase (deg)", -180.0, 180.0, 0.0, 1.0)
        duty = st.slider("Duty cycle (square)", 1, 99, 50) if sig_type=="square" else 50
        f1 = st.slider("Chirp f_start (Hz)", 1.0, max(10.0, fs/2), 100.0, 1.0) if sig_type=="chirp" else None
        f2 = st.slider("Chirp f_end (Hz)", 1.0, max(10.0, fs/2), 1000.0, 1.0) if sig_type=="chirp" else None
    with col2:
        t, x = generate_signal(sig_type, fs, duration, amp=amp, f=freq, phase_deg=phase, duty=duty, f_start=f1, f_end=f2)
        if noise_power > 0:
            x = x + np.random.normal(scale=np.sqrt(noise_power), size=x.shape)
        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(t, x)
        ax.set_title(f"Time Domain: {sig_type}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

with alias_tab:
    st.subheader("Sampling & Aliasing")
    c1, c2 = st.columns(2)
    with c1:
        f_true = st.slider("True sinusoid frequency (Hz)", 0.0, 5*fs, min(1.5*fs, 1000.0), 1.0)
    with c2:
        fs_alias = st.slider("Sampling rate for demo (Hz)", 100, 50000, fs, 100)
    t2, x2 = generate_signal("sine", fs_alias, duration, amp=1.0, f=f_true)
    k = np.round(f_true/fs_alias)
    f_alias = abs(f_true - k*fs_alias)

    c3, c4 = st.columns(2)
    with c3:
        fig1, ax1 = plt.subplots(figsize=(7,3))
        ax1.plot(t2, x2)
        ax1.set_title(f"Sampled Sine at fs={fs_alias} Hz")
        ax1.set_xlabel("t (s)")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1, clear_figure=True)
    with c4:
        freqs, X = fft_analyze(x2, fs_alias, nfft=nfft, window_name=window_name)
        fig2, ax2 = plt.subplots(figsize=(7,3))
        ax2.plot(freqs, 20*np.log10(np.maximum(np.abs(X), 1e-12)))
        ax2.set_title(f"|FFT| dB — alias ≈ {f_alias:.2f} Hz")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)

with fft_tab:
    st.subheader("FFT & Windowing")
    cols = st.columns(2)
    with cols[0]:
        sig_type2 = st.selectbox("Signal type (FFT)", available_signals(), index=0)
        f_fft = st.slider("Signal frequency (Hz)", 0.0, fs/2, min(500.0, fs/2), 1.0)
        frac_cycles = st.checkbox("Use fractional cycles (introduce leakage)", value=True)
    with cols[1]:
        t3, x3 = generate_signal(sig_type2, fs, duration, amp=1.0, f=f_fft)
        if frac_cycles:
            t_temp, x_temp = generate_signal(sig_type2, fs, duration*1.013, amp=1.0, f=f_fft)
            x3 = x_temp[:len(t3)]
        if noise_power > 0:
            x3 = x3 + np.random.normal(scale=np.sqrt(noise_power), size=x3.shape)
        freqs3, X3 = fft_analyze(x3, fs, nfft=nfft, window_name=window_name)

        figt, axt = plt.subplots(figsize=(7,3))
        axt.plot(t3, x3)
        axt.set_title("Time Domain")
        axt.set_xlabel("t (s)")
        axt.grid(True, alpha=0.3)
        st.pyplot(figt, clear_figure=True)

        figf, axf = plt.subplots(figsize=(7,3))
        axf.plot(freqs3, 20*np.log10(np.maximum(np.abs(X3), 1e-12)))
        axf.set_title(f"FFT with {window_name} window (N={nfft})")
        axf.set_xlabel("Frequency (Hz)")
        axf.set_ylabel("Magnitude (dB)")
        axf.grid(True, alpha=0.3)
        st.pyplot(figf, clear_figure=True)

with filt_tab:
    st.subheader("Filtering (FIR & IIR)")
    c1, c2 = st.columns(2)
    with c1:
        ftype = st.selectbox("Filter type", ["FIR", "IIR (Butterworth)"])
        band = st.selectbox("Band", ["lowpass", "highpass", "bandpass", "bandstop"], index=0)
        if band in ("lowpass", "highpass"):
            fcut = st.slider("Cutoff (Hz)", 10.0, fs/2 - 10.0, min(500.0, fs/2 - 10.0), 10.0)
            fcut2 = None
        else:
            fcut = st.slider("f1 (Hz)", 10.0, fs/2 - 20.0, min(200.0, fs/2 - 20.0), 10.0)
            fcut2 = st.slider("f2 (Hz)", fcut+10.0, fs/2 - 1.0, min(2000.0, fs/2 - 1.0), 10.0)
        order = st.slider("Order", 2, 200 if ftype=="FIR" else 10, 8 if ftype!="FIR" else 51)
        test_sig_f = st.slider("Test signal freq (Hz)", 0.0, fs/2, min(2000.0, fs/2), 1.0)

    with c2:
        t4, x4 = generate_signal("sine", fs, duration, amp=1.0, f=test_sig_f)
        if ftype == "FIR":
            b = design_fir(band, fs, order, fcut, fcut2)
            a = np.array([1.0])
        else:
            b, a = design_iir(band, fs, order, fcut, fcut2)
        y4 = apply_filter(b, a, x4)

        w, H = freqz_response(b, a, worN=2048, fs=fs)

        fig1, ax1 = plt.subplots(figsize=(7,3))
        ax1.semilogx(np.maximum(w, 1e-6), 20*np.log10(np.maximum(np.abs(H), 1e-12)))
        ax1.set_title("Filter |H(f)| (dB)")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which='both', alpha=0.3)
        st.pyplot(fig1, clear_figure=True)

        fig2, ax2 = plt.subplots(figsize=(7,3))
        ax2.plot(t4, x4, label='input')
        ax2.plot(t4, y4, label='output')
        ax2.set_title("Time Response")
        ax2.set_xlabel("t (s)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)

with conv_tab:
    st.subheader("Convolution Demo")
    colA, colB = st.columns(2)
    with colA:
        s1_type = st.selectbox("Signal A", available_signals(), key='conv_a')
        fA = st.slider("f_A (Hz)", 0.0, fs/2, min(200.0, fs/2), 1.0)
        tA, xA = generate_signal(s1_type, fs, duration/5, amp=1.0, f=fA)
        figA, axA = plt.subplots(figsize=(7,2.5))
        axA.plot(tA, xA)
        axA.set_title("x[n]")
        axA.grid(True, alpha=0.3)
        st.pyplot(figA, clear_figure=True)
    with colB:
        s2_type = st.selectbox("Signal B (acts like h[n])", ["impulse", "rectangle", "exponential", "sine"], key='conv_b')
        fB = st.slider("f_B (Hz) (for sine)", 0.0, fs/2, min(50.0, fs/2), 1.0)
        tB = np.arange(0, int(fs*duration/10)) / fs
        if s2_type == 'impulse':
            h = np.zeros_like(tB); h[0] = 1.0
        elif s2_type == 'rectangle':
            L = max(1, int(0.02*fs))
            h = np.zeros_like(tB)
            h[:L] = 1
        elif s2_type == 'exponential':
            tau = 0.02
            h = np.exp(-tB/tau)
        else:
            h = np.sin(2*np.pi*fB*tB)
        y = convolve_signals(xA, h)
        figB, axB = plt.subplots(figsize=(7,2.5))
        axB.plot(np.arange(len(y))/fs, y)
        axB.set_title("y[n] = x[n]*h[n]")
        axB.grid(True, alpha=0.3)
        st.pyplot(figB, clear_figure=True)

with spec_tab:
    st.subheader("Spectrogram")
    sp1, sp2 = st.columns(2)
    with sp1:
        sweep = st.checkbox("Use chirp sweep", value=True)
        if sweep:
            f_start = st.slider("Chirp start (Hz)", 1.0, fs/2-10, 50.0)
            f_end = st.slider("Chirp end (Hz)", f_start+1, fs/2-1, min(fs/2-1, 2000.0))
            t5, x5 = generate_signal("chirp", fs, duration, amp=1.0, f_start=f_start, f_end=f_end)
        else:
            fS = st.slider("Single tone (Hz)", 0.0, fs/2, min(500.0, fs/2))
            t5, x5 = generate_signal("sine", fs, duration, amp=1.0, f=fS)
        if noise_power > 0:
            x5 = x5 + np.random.normal(scale=np.sqrt(noise_power), size=x5.shape)
    with sp2:
        figS = plot_spectrogram(x5, fs)
        st.pyplot(figS, clear_figure=True)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, NumPy, SciPy and Matplotlib.")
