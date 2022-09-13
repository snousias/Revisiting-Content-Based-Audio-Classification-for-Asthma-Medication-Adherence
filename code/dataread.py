"""
MIT License

Copyright (c) 2022 Visualization and Virtual Reality Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import wave
import os
import glob
import scipy.io.wavfile as wv
import scipy
from keras.models import Sequential,Model,load_model,save_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization,Flatten,Conv2D,MaxPooling2D,Input,LeakyReLU,ReLU,Softmax
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from sklearn.model_selection import ShuffleSplit,train_test_split,KFold
from config import path
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.signal import get_window
import copy

def create_cepstrum(X,hop=32,wlen=128):
    for i in range(0,len(X),hop):
        if (i+wlen)<len(X):
            signal = X[i: i+wlen]
            ceps = real_cepstrum(signal, None)
            ceps = np.expand_dims(ceps, axis=1)
            if i==0:
                feats=ceps
            else:
                feats=np.concatenate((feats, ceps), axis=1)
    return feats

def real_cepstrum(x, n=None):
    spectrum = np.fft.fft(x, n=n)
    spectrum=np.abs(spectrum)
    #spectrum=spectrum[:(len(spectrum)/2)]
    #plt.plot(spectrum)
    #plt.show()
    ceps=np.log(spectrum+1)
    ceps = np.fft.ifft(ceps).real
    ceps=ceps[:int(len(ceps)/2)]
    #plt.plot(ceps)
    #plt.show()
    return ceps

def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data

def create_spectrogram(ts,NFFT,noverlap = None):
    def get_xn(Xs, n):
        '''
        calculate the Fourier coefficient X_n of
        Discrete Fourier Transform (DFT)
        '''
        L = len(Xs)
        ks = np.arange(0, L, 1)
        xn = np.sum(Xs * np.exp((1j * 2 * np.pi * ks * n) / L)) / L
        return (xn)

    def get_xns(ts):
        '''
        Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
        and multiply the absolute value of the Fourier coefficients by 2,
        to account for the symetry of the Fourier coefficients above the Nyquest Limit.
        '''
        mag = []
        L = len(ts)
        for n in range(int(L / 2)):  # Nyquest Limit
            mag.append(np.abs(get_xn(ts, n)) * 2)
        return (mag)

    '''
          ts: original time series
        NFFT: The number of data points used in each block for the DFT.
          Fs: the number of points sampled per second, so called sample_rate
    noverlap: The number of points of overlap between blocks. The default value is 128.
    '''
    if noverlap is None:
        noverlap = NFFT/2
    noverlap = int(noverlap)
    starts  = np.arange(0,len(ts),NFFT-noverlap,dtype=int)
    # remove any window with less than NFFT sample size
    starts  = starts[starts + NFFT < len(ts)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = get_xns(ts[start:start + NFFT])
        xns.append(ts_window)
    spec = np.array(xns).T
    # rescale the absolute value of the spectrogram as rescaling is standard
    # spec = 10*np.log10(spec)
    assert spec.shape[1] == len(starts)
    return (starts,spec)

def create_cepstrogram(ts,NFFT,noverlap = None):
    def get_xn(Xs, n):
        '''
        calculate the Fourier coefficient X_n of
        Discrete Fourier Transform (DFT)
        '''
        L = len(Xs)
        ks = np.arange(0, L, 1)
        xn = np.sum(Xs * np.exp((1j * 2 * np.pi * ks * n) / L)) / L
        return (xn)

    def get_xns(ts):
        '''
        Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
        and multiply the absolute value of the Fourier coefficients by 2,
        to account for the symetry of the Fourier coefficients above the Nyquest Limit.
        '''
        mag = []
        L = len(ts)
        for n in range(int(L / 2)):  # Nyquest Limit
            mag.append(np.abs(get_xn(ts, n)) * 2)
        return (mag)

    '''
          ts: original time series
        NFFT: The number of data points used in each block for the DFT.
          Fs: the number of points sampled per second, so called sample_rate
    noverlap: The number of points of overlap between blocks. The default value is 128.
    '''
    if noverlap is None:
        noverlap = NFFT/2
    noverlap = int(noverlap)
    starts  = np.arange(0,len(ts),NFFT-noverlap,dtype=int)
    # remove any window with less than NFFT sample size
    starts  = starts[starts + NFFT < len(ts)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        audioDataWindow=ts[start:start + NFFT]+pow(10, -10)

        powerspectrum = np.abs(np.fft.fft(audioDataWindow)) ** 2
        powerspectrum=replaceZeroes(powerspectrum)
        cepstrum = np.fft.ifft(np.log(powerspectrum))
        cepstrum=np.abs(cepstrum) ** 2
        cepstrum=cepstrum[:int(NFFT/2)]

        # ts_window = get_xns(ts[start:start + NFFT])
        # ts_window = np.power(ts_window,2)
        # ts_padded=np.zeros((NFFT,))
        # ts_padded[:np.shape(ts_window)[0]]=ts_window
        # ts_liftered=np.fft.ifft(ts_padded)/(NFFT/2)
        # # ts_liftered = np.asarray(get_xns(ts_padded))/(NFFT/2)
        # ts_liftered = np.power(np.abs(ts_liftered),2)
        # ts_liftered_list=ts_liftered.tolist()

        xns.append(cepstrum.tolist())
    spec = np.array(xns).T
    # rescale the absolute value of the spectrogram as rescaling is standard
    # spec = 10*np.log10(specX)
    assert spec.shape[1] == len(starts)
    return (starts,spec)

def mfcc(X):

    FFT_size = 128
    sample_rate=8000
    mel_filter_num = 40
    hop_size = 32

    def frame_audio(audio, FFT_size=512, hop_size=128, sample_rate=8000):
        # hop_size in ms

        audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
        # frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_len = hop_size

        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, FFT_size))

        for n in range(frame_num):
            frames[n] = audio[n * frame_len:n * frame_len + FFT_size]

        return frames

    def freq_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
        fmin_mel = freq_to_mel(fmin)
        fmax_mel = freq_to_mel(fmax)

        # print("MEL min: {0}".format(fmin_mel))
        # print("MEL max: {0}".format(fmax_mel))

        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
        freqs = met_to_freq(mels)

        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    def get_filters(filter_points, FFT_size):
        filters = np.zeros((len(filter_points) - 2, int(FFT_size / 2 + 1)))
        for n in range(len(filter_points) - 2):
            filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1,
                                                                             filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0,
                                                                                 filter_points[n + 2] - filter_points[
                                                                                     n + 1])
        return filters

    def dct(dct_filter_num, filter_len):
        basis = np.empty((dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)
        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        return basis




    window = get_window("hann", FFT_size, fftbins=True)
    audio_framed = frame_audio(X, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
    audio_win = audio_framed * window
    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
    audio_fft = np.transpose(audio_fft)
    audio_power = np.square(np.abs(audio_fft))
    # print(audio_power.shape)
    freq_min = 0
    freq_high = sample_rate / 2

    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=sample_rate)
    filters = get_filters(filter_points, FFT_size)
    enorm = 2.0 / (mel_freqs[2:mel_filter_num + 2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]
    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered+1)
    audio_log.shape
    dct_filter_num = 32

    dct_filters = dct(dct_filter_num, mel_filter_num)

    cepstral_coefficents = np.dot(dct_filters, audio_log)
    # cepstral_coefficents.shape
    return cepstral_coefficents

def getDataSets(filenames,annotation,classes):
    train_fnms, test_fnms=next(iter(KFold(10, True).split(filenames)))

    # Trainset
    train_files = [filenames[i] for i in train_fnms]
    X, Y = fetchDataAndLabelsCNN(train_files, annotation, classes)
    data_train = np.asarray(X)
    label_train = to_categorical(Y)
    # data_train.resize(len(data_train), 50, 40, 1)
    data_train.resize(len(data_train), 250, 16, 1)

    # Testset
    test_files = [filenames[i] for i in test_fnms]
    X_test, Y_test = fetchDataAndLabelsCNN(test_files, annotation, classes)
    data_test = np.asarray(X_test)
    label_test = to_categorical(Y_test)
    # data_test.resize(len(data_test), 50, 40, 1)
    data_test.resize(len(data_test), 250, 16, 1)
    return data_train,label_train,data_test,label_test

def fetchAnnotatedFilenames(selection=['f1','g1','a1']):
    f1=glob.glob(os.path.join(path+'_f1/', '*.wav'))
    annotation_f1 = np.genfromtxt(path + '_f1/' + 'annotation.csv', delimiter=',', dtype='<U50')

    g1=glob.glob(os.path.join(path+'_g1/', '*.wav'))
    annotation_g1 = np.genfromtxt(path + '_g1/' + 'annotation.csv', delimiter=',', dtype='<U50')

    a1=glob.glob(os.path.join(path+'_a1/', '*.wav'))
    annotation_a1=np.genfromtxt(path+'_a1/'+'annotation.csv', delimiter=',',dtype='<U50')

    a0=glob.glob(os.path.join(path+'_a0/', '*.wav'))
    annotation_a0=np.genfromtxt(path+'_a0/'+'annotation.csv', delimiter=',',dtype='<U50')

    filenames=[]
    if 'f1' in selection:
        filenames.extend(f1)
    if 'g1' in selection:
        filenames.extend(g1)
    if 'a1' in selection:
        filenames.extend(a1)
    if 'a0' in selection:
        filenames.extend(a0)

    annotation=np.empty((0,4))
    if 'f1' in selection:
        annotation = np.concatenate((annotation, annotation_f1))
    if 'g1' in selection:
        annotation = np.concatenate((annotation, annotation_g1))
    if 'a1' in selection:
        annotation = np.concatenate((annotation, annotation_a1))
    if 'a0' in selection:
        annotation = np.concatenate((annotation, annotation_a0))


    # annotation = np.concatenate((annotation_f1, annotation_g1, annotation_a1))
    # filenames=a0
    # annotation=annotation_a0


    return filenames,annotation

def fetchDataAndLabelsCNN(train_files,annotation,classes,step=1,doResize=True,loadAnnotatedSubsetOnly=True):
    Y  = []
    X  = []
    Ym = []
    Xm = []
    for filename in train_files:
        #print(filename)
        w = wv.read(filename)
        w_ = wave.openfp(filename)
        maxval = 2 ** ((w_.getsampwidth() * (8) - 8))
        fs = w[0]
        audioData = w[1]
        audioData = audioData / maxval
        xmean=np.mean(audioData)
        xstd=np.std(audioData)
        audioData = (audioData - xmean)/xstd
        audioData=audioData/np.max(audioData)
        audioData = (audioData + 1) / 2
        step=1
        patchSize=4000
        # audioData=audioData[1::step]
        audioDataTemp=[]
        for i in range(len(audioData)):
            for j in range(step):
                audioDataTemp.append(audioData[i])
        audioData=np.asarray(audioDataTemp)
        head_tail = os.path.split(filename)
        fwe = head_tail[1].split(sep='.')[0]

        if not loadAnnotatedSubsetOnly:
            annotatedAudio=np.full((np.shape(audioData)[0],),3)
            A = annotation[annotation[:, 0] == head_tail[1], :]
            for ind,row in enumerate(A):
                mLabel = row[1]
                startA = int(int(row[2]))
                stopA = int(int(row[3]))+1
                c = classes.index(mLabel)
                annotatedAudio[startA:stopA]=c
            startIndices=list(range(0, np.shape(audioData)[0] - patchSize+1, 100))
            for a in startIndices:
                datinput = audioData[a:a + patchSize]
                dat = np.asarray(datinput)
                dat.resize((1, 250, 16))
                c = annotatedAudio[a+int(patchSize/2)]
                Ym.append(c)
                Xm.append(dat)

        if loadAnnotatedSubsetOnly:
            A = annotation[annotation[:, 0] == head_tail[1], :]
            for ind,row in enumerate(A):
                mLabel = row[1]
                startA = int(int(row[2]))
                stopA = int(int(row[3]))+1
                # startA = int(int(row[2]) / step)
                # stopA = int(int(row[3]) / step)
                c = classes.index(mLabel)
                startIndices = list(range(startA, (stopA - patchSize)+1, 50))
                # print(startIndices)
                # print(row)
                for i in startIndices:
                    Y.append(c)
                    # print(c)
                    datinput=audioData[i:i + patchSize]
                    # datinput = datinput[1::step]
                    dat = np.asarray(datinput)
                    # dat=dat*127
                    # dat=np.round(dat * 2 ** 0).astype(np.int8)
                    # fnm=fwe+"_"+str(ind)+"_"+mLabel+".csv"
                    # np.savetxt("E:/Dropbox/_GroundWork/DeepCNNSparseCoding/output/"+fnm, dat, delimiter=",",fmt='%1.5f')
                    # dat= (dat+np.ones(np.shape(dat)))/2
                    dat.resize((1, 250, 16))
                    # dat.resize((1, 50, 40))
                    X.append(dat)
                    # X=np.concatenate((X,dat),axis=0)
                    # print('ok')






    if loadAnnotatedSubsetOnly:
        return X,Y
    else:
        return Xm, Ym








def fetchDataAndLabelsML(train_files,annotation,classes,type,step=1,doResize=True,loadAnnotatedSubsetOnly=True):
    Y = []
    X = []
    Xm= []
    Ym= []
    for filename in train_files:
        w = wv.read(filename)
        w_ = wave.openfp(filename)
        ff=w_.getsampwidth()

        maxval = 2 ** ((w_.getsampwidth() * 8 - 8))
        fs = w[0]
        audioData = w[1]
        audioData = audioData / maxval

        # xmean=np.mean(audioData)
        # xstd=np.std(audioData)
        # audioData = (audioData - xmean)/xstd
        # audioData = (audioData + 1) / 2
        # patchSize=4000
        # hop=500
        # if step>1:
        #     step = 2
        #     audioData=audioData[1::step]
        #     audioDataTemp=[]
        #     for i in range(len(audioData)):
        #         for j in range(step):
        #             audioDataTemp.append(audioData[i])
        #     audioData=np.asarray(audioDataTemp)
        # audioData=audioData/np.max(audioData)

        nperseg = 128
        noverlap = 3 * 32
        window=int(fs/2)
        hop=nperseg-noverlap
        if type=="spect":
            freqs, times, input = scipy.signal.spectrogram(audioData,fs,nfft=512,nperseg=nperseg,noverlap=noverlap)
        if type=="cepst":
            input = create_cepstrum(audioData)
        if type == "mfcc":
            input = mfcc(audioData)


        # plt.figure()
        # plt_spec = plt.imshow(spec)
        # plt.show()
        # plt.figure()
        # plt_spec = plt.imshow(spec, origin='lower')
        # plt.show()
        # # exit(50)
        # spec = mfcc(audioData)
        # plt.figure()
        # plt_spec = plt.imshow(spec, origin='lower')
        # plt.show()
        # exit(50)
        # mfcctemp=[]
        # for i in range(0,len(audioData)-512,128):
        #     Xin=audioData[i:i+512]
        #     spectrogram=mfcc(Xin)
        #     mfcctemp.append(spectrogram)
        # mfccs=np.vstack(mfcctemp)


        # Mixed
        if not loadAnnotatedSubsetOnly:
            head_tail = os.path.split(filename)
            fwe = head_tail[1].split(sep='.')[0]
            annotatedAudio = np.full((np.shape(audioData)[0],), 3)
            A = annotation[annotation[:, 0] == head_tail[1], :]
            for ind, row in enumerate(A):
                mLabel = row[1]
                startA = int(int(row[2]))
                stopA = int(int(row[3])) + 1
                c = classes.index(mLabel)
                annotatedAudio[startA:stopA] = c
            startIndices = list(range(0, np.shape(audioData)[0] - window + 1, 100))
            for a in startIndices:
                c = annotatedAudio[a+int(window/2)]
                corresponding_window_A = int(np.floor(a / hop))
                corresponding_window_B = int(np.floor(((a+window) - 3 * hop) / hop))
                featInput = input[:, corresponding_window_A:corresponding_window_B]
                indices = np.asarray(range(0, np.shape(featInput)[0], np.floor(np.shape(featInput)[0] / 32).astype(int)))
                feat = featInput[indices,]
                feat = np.mean(feat, axis=1)
                if np.isfinite(feat).all():
                    Ym.append(c)
                    Xm.append(feat)

        # Ummixed
        if loadAnnotatedSubsetOnly:
            head_tail = os.path.split(filename)
            fwe=head_tail[1].split(sep='.')[0]
            A = annotation[annotation[:, 0] == head_tail[1], :]
            for ind,row in enumerate(A):
                mLabel = row[1]
                startA = int(int(row[2]))
                stopA = int(int(row[3]))+1
                corresponding_window_A= int(np.floor(startA/hop))
                corresponding_window_B = int(np.floor((stopA-3*hop) / hop))
                if ((corresponding_window_A != corresponding_window_B) & (corresponding_window_A >= 0) & (corresponding_window_B <= (np.shape(input)[1]))):
                    featInput=input[:,corresponding_window_A:corresponding_window_B]
                    indices=np.asarray(range(0, np.shape(featInput)[0], np.floor(np.shape(featInput)[0] / 32).astype(int)))
                    feat=featInput[indices,]
                    #feat=np.mean(featInput[:256].reshape(-1,8 ), 1)
                    feat = np.mean(feat, axis=1)
                    #plt.plot(feat)
                    #plt.show()
                    c = classes.index(mLabel)
                    #plt.plot(feat)
                    #plt.show()
                    if np.isfinite(feat).all():
                        Y.append(c)
                        X.append(feat)

    if loadAnnotatedSubsetOnly:
        return X,Y
    else:
        return Xm,Ym







def fetchDataAndLabelsLSTM(train_files,annotation,classes,type,step=1,doResize=True,loadAnnotatedSubsetOnly=True):
    Y = []
    X = []
    initialized = False

    Ym= []
    Xm = []
    initializedm = False


    for filename in train_files:
        #print(filename)
        w = wv.read(filename)
        w_ = wave.openfp(filename)
        ff=w_.getsampwidth()
        maxval = 2 ** ((w_.getsampwidth() * 8 - 8))
        fs = w[0]
        audioData = w[1]
        audioData = audioData / maxval
        # xmean=np.mean(audioData)
        # xstd=np.std(audioData)
        # audioData = (audioData - xmean)/xstd
        # audioData = (audioData + 1) / 2
        # patchSize=4000
        # hop=500
        # if step>1:
        #     step = 2
        #     audioData=audioData[1::step]
        #     audioDataTemp=[]
        #     for i in range(len(audioData)):
        #         for j in range(step):
        #             audioDataTemp.append(audioData[i])
        #     audioData=np.asarray(audioDataTemp)
        # audioData=audioData/np.max(audioData)
        nperseg = 512
        noverlap = int(nperseg*3/4)
        window = 15
        hop=nperseg-noverlap
        if type=="spect":
            freqs, times, input = scipy.signal.spectrogram(audioData,fs,nfft=512,nperseg=nperseg,noverlap=noverlap)
        if type=="cepst":
            input = create_cepstrum(audioData)
        if type == "mfcc":
            input = mfcc(audioData)
        # plt.figure()
        # plt_spec = plt.imshow(spec)
        # plt.show()
        # plt.figure()
        # plt_spec = plt.imshow(spec, origin='lower')
        # plt.show()
        # # exit(50)
        # spec = mfcc(audioData)
        # plt.figure()
        # plt_spec = plt.imshow(spec, origin='lower')
        # plt.show()
        # exit(50)
        # mfcctemp=[]
        # for i in range(0,len(audioData)-512,128):
        #     Xin=audioData[i:i+512]
        #     spectrogram=mfcc(Xin)
        #     mfcctemp.append(spectrogram)
        # mfccs=np.vstack(mfcctemp)

        if not loadAnnotatedSubsetOnly:
            head_tail = os.path.split(filename)
            annotatedSpectrum = np.full((np.shape(input)[1],), 3)
            A = annotation[annotation[:, 0] == head_tail[1], :]
            for ind, row in enumerate(A):
                mLabel = row[1]
                startA = int(int(row[2]))
                stopA = int(int(row[3])) + 1
                c = classes.index(mLabel)
                corresponding_window_A= int(np.floor(startA/hop))
                corresponding_window_B = int(np.floor((stopA-3*hop) / hop))
                annotatedSpectrum[corresponding_window_A:corresponding_window_B] = c
            featInput = input
            indices = np.asarray(range(0, np.shape(featInput)[0], np.floor(np.shape(featInput)[0] / 32).astype(int)))
            feat = featInput[indices,]
            startIndices = list(range(0, np.shape(annotatedSpectrum)[0] - window + 1, 5))
            for a in startIndices:
                if a + window <= (np.shape(feat)[1]):
                    F = np.transpose(feat[:, a:a + window])
                    if np.isfinite(F).all():
                        Ym.append(annotatedSpectrum[a+int(window/2)])
                        Xm.append(F)
                        # if not initializedm:
                        #     F = np.expand_dims(F, axis=0)
                        #     Xm = copy.deepcopy(F)
                        #     initializedm = True
                        # else:
                        #     Xm = np.concatenate((Xm, np.expand_dims(F, axis=0)), axis=0)

        if loadAnnotatedSubsetOnly:
            head_tail = os.path.split(filename)
            fwe=head_tail[1].split(sep='.')[0]
            A = annotation[annotation[:, 0] == head_tail[1], :]
            for ind,row in enumerate(A):
                mLabel = row[1]
                startA = int(int(row[2]))
                stopA = int(int(row[3]))+1
                corresponding_window_A= int(np.floor(startA/hop))
                corresponding_window_B = int(np.floor((stopA-3*hop) / hop))
                if ((corresponding_window_A != corresponding_window_B) & (corresponding_window_A >= 0) & (corresponding_window_B <= (np.shape(input)[1]))):
                    featInput=input[:,corresponding_window_A:corresponding_window_B]
                    indices=np.asarray(range(0, np.shape(featInput)[0], np.floor(np.shape(featInput)[0] / 32).astype(int)))
                    feat=featInput[indices,]
                    C = classes.index(mLabel)
                    for i in range(0,np.shape(feat)[0]-window,2):
                        if i+window<=(np.shape(feat)[1]):
                            F=np.transpose(feat[:,i:i+window])

                            if np.isfinite(F).all():
                                Y.append(C)
                                X.append(F)

                                # if not initialized:
                                #     F=np.expand_dims(F,axis=0)
                                #     X=copy.deepcopy(F)
                                #     initialized=True
                                # else:
                                #     X=np.concatenate((X,np.expand_dims(F,axis=0)),axis=0)
    if loadAnnotatedSubsetOnly:
        X=np.asarray(X)
        return X,Y
    else:
        Xm=np.asarray(Xm)
        return Xm,Ym




