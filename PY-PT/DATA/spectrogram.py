import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(1, '..\\')
import utils
import scipy.signal as scs
import scipy.fftpack
# Perform spectrogram on unfiltered data


def spectro(data):
    fig = plt.figure()
    fig.set_size_inches(6, 5)
    i0 = 950
    i1 = len(data['time'])-2500
    time = np.array(data['time'][i0:i1]) - data['time'][i0]
    signal = np.array(scs.detrend(np.array(data['p_comb'][i0:i1])))


    #ax_top = fig.add_subplot(211)
    ax_spec = fig.add_subplot(111)

    #ax_top.plot(time-time[0], signal)

    Pxx, freqs, bins, cax = ax_spec.specgram(signal, NFFT=128, Fs=1/(time[1]-time[0]), noverlap=128 / 2, mode='magnitude', pad_to=256, cmap=plt.get_cmap('inferno'))
    ax_spec.set_xlabel('Time (s)')
    ax_spec.set_ylabel('Frequency (Hz)')


    # Do fft on signal
    # Number of samplepoints
    N = len(signal)
    # sample spacing
    T = (time[1]-time[0])
    x = np.linspace(0.0, N*T, N)
    y = signal
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), '-k', linewidth=1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.tick_params(direction='in')
    fig.set_size_inches(6, 5)
    plt.show()



if __name__=='__main__':
    data = utils.read_json('data/data_raw.json')

    spectro(data)