import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.title("📡 Interactive DSP Visualizer")

# Sidebar settings
st.sidebar.header("Signal Settings")

signal_type = st.sidebar.selectbox(
    "Signal Type",
    ["Sine","Square","Triangle","Sawtooth","Chirp","White Noise","Impulse","Step"]
)

freq = st.sidebar.slider("Frequency (Hz)",1,1000,50)
fs = st.sidebar.slider("Sampling Frequency",500,5000,1000)
duration = st.sidebar.slider("Duration (s)",1,5,2)

t = np.linspace(0,duration,int(fs*duration),endpoint=False)

# Signal Generator
if signal_type=="Sine":
    x=np.sin(2*np.pi*freq*t)

elif signal_type=="Square":
    x=signal.square(2*np.pi*freq*t)

elif signal_type=="Triangle":
    x=signal.sawtooth(2*np.pi*freq*t,0.5)

elif signal_type=="Sawtooth":
    x=signal.sawtooth(2*np.pi*freq*t)

elif signal_type=="Chirp":
    x=signal.chirp(t,f0=1,f1=freq,t1=duration)

elif signal_type=="White Noise":
    x=np.random.normal(0,1,len(t))

elif signal_type=="Impulse":
    x=np.zeros(len(t))
    x[0]=1

elif signal_type=="Step":
    x=np.ones(len(t))

# DSP module selection
module = st.sidebar.selectbox(
    "DSP Module",
    [
        "Signal Generator",
        "DFT Spectrum",
        "FFT Spectrum",
        "Butterworth Filter",
        "Chebyshev Filter",
        "Spectrogram",
        "Convolution",
        "Window Functions"
    ]
)

# SIGNAL PLOT
if module=="Signal Generator":

    fig,ax=plt.subplots()
    ax.plot(t,x)
    ax.set_title("Generated Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


# DFT
elif module=="DFT Spectrum":

    N=len(x)
    X=np.zeros(N,dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k]+=x[n]*np.exp(-2j*np.pi*k*n/N)

    freqs=np.arange(N)*(fs/N)

    fig,ax=plt.subplots()
    ax.plot(freqs[:N//2],np.abs(X[:N//2]))
    ax.set_title("DFT Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    st.pyplot(fig)


# FFT
elif module=="FFT Spectrum":

    X=np.fft.fft(x)
    freqs=np.fft.fftfreq(len(X),1/fs)

    fig,ax=plt.subplots()
    ax.plot(freqs[:len(freqs)//2],np.abs(X[:len(X)//2]))
    ax.set_title("FFT Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    st.pyplot(fig)


# BUTTERWORTH FILTER
elif module=="Butterworth Filter":

    filter_type = st.selectbox(
        "Filter Type",
        ["Lowpass","Highpass","Bandpass","Bandstop"]
    )

    if filter_type in ["Lowpass","Highpass"]:

        cutoff = st.slider("Cutoff Frequency",1,fs//2,100)
        b,a = signal.butter(4,cutoff/(fs/2),btype=filter_type.lower())

    else:

        low = st.slider("Low Frequency",1,fs//4,50)
        high = st.slider("High Frequency",fs//4,fs//2,200)

        if filter_type=="Bandpass":
            b,a=signal.butter(4,[low/(fs/2),high/(fs/2)],btype='bandpass')

        else:
            b,a=signal.butter(4,[low/(fs/2),high/(fs/2)],btype='bandstop')

    y=signal.lfilter(b,a,x)

    fig,ax=plt.subplots()
    ax.plot(t,x,label="Input")
    ax.plot(t,y,label="Filtered")
    ax.legend()
    ax.set_title("Butterworth Filter Output")
    st.pyplot(fig)


# CHEBYSHEV FILTER
elif module=="Chebyshev Filter":

    filter_type = st.selectbox(
        "Filter Type",
        ["Lowpass","Highpass","Bandpass","Bandstop"]
    )

    ripple = st.slider("Passband Ripple (dB)",1,5,1)

    if filter_type in ["Lowpass","Highpass"]:

        cutoff = st.slider("Cutoff Frequency",1,fs//2,100)
        b,a = signal.cheby1(4,ripple,cutoff/(fs/2),btype=filter_type.lower())

    else:

        low = st.slider("Low Frequency",1,fs//4,50)
        high = st.slider("High Frequency",fs//4,fs//2,200)

        if filter_type=="Bandpass":
            b,a=signal.cheby1(4,ripple,[low/(fs/2),high/(fs/2)],btype='bandpass')

        else:
            b,a=signal.cheby1(4,ripple,[low/(fs/2),high/(fs/2)],btype='bandstop')

    y=signal.lfilter(b,a,x)

    fig,ax=plt.subplots()
    ax.plot(t,x,label="Input")
    ax.plot(t,y,label="Filtered")
    ax.legend()
    ax.set_title("Chebyshev Filter Output")
    st.pyplot(fig)


# SPECTROGRAM
elif module=="Spectrogram":

    fig,ax=plt.subplots()
    ax.specgram(x,Fs=fs)
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


# CONVOLUTION
elif module=="Convolution":

    h=np.ones(20)/20
    y=np.convolve(x,h,mode='same')

    fig,ax=plt.subplots()
    ax.plot(t,x,label="Input")
    ax.plot(t,y,label="Convolved")
    ax.legend()
    st.pyplot(fig)


# WINDOW FUNCTIONS
elif module=="Window Functions":

    window_type = st.selectbox(
        "Window Type",
        ["hann","hamming","blackman"]
    )

    window = signal.get_window(window_type,len(x))

    fig,ax=plt.subplots()
    ax.plot(window)
    ax.set_title(window_type+" Window")
    st.pyplot(fig)
