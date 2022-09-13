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
import wave
import os
import scipy.io.wavfile as wv
from sklearn.metrics import confusion_matrix
import random
from config import *
from dataread import fetchAnnotatedFilenames
import pywt
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage



def decision_t_2014(X):
    Y_t = []
    Y_p = []
    for i in range(len(X)):
        for j in range(0, len(X[0]) - 1000, 1000):
            step = 100
            # print("nikos")
            coef_mat = []
            # wavelet = 'cmor1.5-1.0'
            sst = X[i, j:j+1000]
            for k in range(0, np.shape(sst)[0], step):
                w = sst[k: k + step]
                coef, freqs = pywt.cwt(w, np.arange(1.625, 2.00, 0.005), 'morl')
                coef_mat.append(coef)
            cwtmat = np.array(coef_mat)
            frequencies = pywt.scale2frequency('morl', np.arange(1.625, 2.00, 0.005)) * 8000

            # print(cwtmat.shape) # (10, 75, 100)

            thres = 50  # .1*(10**223)

            # print("Frequencies: ", frequencies)
            # print("coefmatshape:", coef.shape)  # (75, 1000) (scale, samples = 8000 * time)
            # print(cwtmat)

            first = np.zeros(np.shape(cwtmat)[0])
            second = np.zeros(np.shape(cwtmat)[0])
            cwt_window = np.zeros(np.shape(cwtmat)[0])
            for k in range(np.shape(cwtmat)[0]):
                cwt_window[k] = np.sum(cwtmat[k, :, :] ** 2)

            '''
            print(cwt_window.shape)
            m = 0
            ccwt = 0
            for i in range(np.shape(cwtmat)[2]):
                ccwt += cwt_window[i]
                m+=1
            ccwt = ccwt/m
            print("cwt: ", ccwt, "Y[2]: ", Y[2])
            '''

            for l in range(np.shape(cwt_window)[0]):
                # print(cwt_window[i])
                if (cwt_window[l] > thres):
                    # print("(first threshold) Potential blister sound, at: ", i, " window, with CWT squared coefficients: ", cwt_window[i])
                    first[l] = 1
                    if l < np.shape(cwt_window)[0] - 4:
                        if cwt_window[l] * 0.75 > cwt_window[l + 4]:
                            second[l] = 1
                    elif l - 4 >= 0:
                        if cwt_window[l] * 0.75 > cwt_window[l - 4]:
                            second[l] = 1
                    else:
                        # 56.0 msec
                        if (cwt_window[l - 4] < cwt_window[l] * 0.75) and (cwt_window[l + 4] < cwt_window[l] * 0.75):
                            second[l] = 1

            y_true = []
            y_pred = []

            for k in range(np.shape(cwt_window)[0]):
                # print(Y[2])
                if first[k] == 1 and second[k] == 1:
                    y_pred = np.append(y_pred, 1)
                    # if Y[k][2] == 1.0:
                    #     y_true = np.append(y_true, 1)
                    #     print('nik');
                    #     break
                    # else:
                    #     y_true = np.append(y_true, 0)
                    #     y_pred = np.append(y_pred, 1)
                    #     break
                else:
                    y_pred = np.append(y_pred, 0)
                    # if Y[k][2] != 1.0:
                    #     print(Y[k][2])
                    #     print('kos')
                    #     continue
                    #     # y_true = np.append(y_true, 1)
                    #     # y_pred = np.append(y_pred, 0)
                    # else:
                    #     y_true = np.append(y_true, 0)
                    #
                    #     break
            # Y_t = np.append(Y_t, y_true)
            Y_p = np.append(Y_p, y_pred)
    return Y_p
    # cm1 = confusion_matrix(Y_t, Y_p)
    # print('Confusion Matrix : \n', cm1)
    #
    # total1 = sum(sum(cm1))
    # #####from confusion matrix calculate accuracy
    # accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    # print('Accuracy : ', accuracy1)
    #
    # sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    # print('Sensitivity : ', sensitivity1)
    #
    # specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    # print('Specificity : ', specificity1)





def ReLU(x):
    return x * (x > 0)


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)




classes = ['Drug', 'Exhale', 'Inhale', 'Noise']

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
filenames, annotation = fetchAnnotatedFilenames()
random.Random(9).shuffle(filenames)
fromScratch = True
doStore = False
p_class = []
t_class = []



scale = np.arange(3.625, 6.0, 0.05)
scale = np.arange(0.825, 16.0, 0.1)
wavelet = 'morl'
sampling_period = 1 / 8000
lim = 448
duration = 20



cmtot = np.zeros((2, 2)).astype(int)
cmanalysis = []
for filename in filenames:
    print(filename)
    w = wv.read(filename)
    w_ = wave.openfp(filename)
    bd=w_.getsampwidth()
    maxval = 2 ** ( (bd* 8)-1)
    fs = w[0]
    audioData = w[1]
    audioData = audioData / maxval

    #ttte=decision_t_2014(audioData)


    # xmean = np.mean(audioData)
    # xstd = np.std(audioData)
    # audioData = (audioData - xmean) / xstd
    # audioData = audioData / np.max(audioData)
    head_tail = os.path.split(filename)
    fwe = head_tail[1].split(sep='.')[0]
    A = annotation[annotation[:, 0] == head_tail[1], :]
    groundtruth = 3 * np.ones(np.shape(audioData))
    for ind, row in enumerate(A):
        mLabel = row[1]
        startA = int(int(row[2]))
        stopA = int(int(row[3]))
        c = classes.index(mLabel)
        groundtruth[startA:stopA] = c
    f = pywt.scale2frequency(wavelet, scale) / sampling_period
    cwtmatr, freqs = pywt.cwt(audioData, scale, wavelet)
    # cwtmatr = ReLU(cwtmatr)
    # im = plt.imshow(cwtmatr, interpolation='nearest', aspect='auto')
    #plt.colorbar(im, orientation='horizontal')
    #plt.show()
    ssq = np.mean(cwtmatr ** 2, axis=0)
    # ssq = ndimage.maximum_filter1d(ssq, 512)
    # ssq = medfilt(ssq, 3)
    # plt.plot(ssq)
    # plt.show()
    # exit(3)
    # output = medfilt(output, 256 + 1)
    # plt.plot(output)
    prediction = np.empty(np.shape(audioData))
    for i in range(0, np.shape(audioData)[0]):
        prediction[i] = 0
    for i in range(500, np.shape(audioData)[0] - 500):
        if ssq[i] > 0.6:
            # prediction[i] = 1
            # if (i+448)<(np.shape(y)[0]-1) & (i-448)<0:
            if np.max(ssq[i + lim:i + (lim + duration)]) < (0.75 * ssq[i]):
                if np.max(ssq[i - (lim + duration):i - lim]) < (0.75 * ssq[i]):
                    prediction[i] = 1
    # prediction = medfilt(prediction, 3)
    fillup = 1000
    for i in range(fillup, np.shape(audioData)[0] - fillup):
        if prediction[i] == 1:
            prediction[i - fillup:i] = 3 * np.ones(np.shape(prediction[i - fillup:i]))
            prediction[i:i + fillup] = 3 * np.ones(np.shape(prediction[i:i + fillup]))

    prediction[prediction == 3] = 1
    prediction = prediction.astype(bool)
    predictionFinal = 3 * np.ones(np.shape(audioData))
    predictionFinal[prediction] = 0

    groundtruthFinal = 3 * np.ones(np.shape(audioData))
    groundtruthFinal[groundtruth==0]=0

    granularity=2000
    step=100
    subsampledGT=[]
    for x in range(0,np.shape(groundtruth)[0]-granularity,step):
        subsampledGT.append(3)
    subsampledGT=np.asarray(subsampledGT)
    for x in range(0,np.shape(subsampledGT)[0]):
        g=groundtruth[step*x:(step*x+granularity)]
        if (np.any(g==0)):
            subsampledGT[x]=0

    subsampledPred=[]
    for x in range(0,np.shape(predictionFinal)[0]-granularity,step):
        subsampledPred.append(3)
    subsampledPred=np.asarray(subsampledPred)
    for x in range(0,np.shape(subsampledPred)[0]):
        g=predictionFinal[step*x:(step*x+granularity)]
        if (np.any(g==0)):
            subsampledPred[x]=0
    # cm = np.asarray(confusion_matrix(subsampledGT, subsampledPred,labels=[0,3]))
    # print(cm)
    # exit(55)
    # decision_t_2014(audioData)
    # plt.plot(ssq)

    plt.plot(groundtruth)
    plt.plot(predictionFinal)
    # plt.plot(subsampledGT)
    # plt.plot(subsampledPred)
    plt.plot(audioData)
    plt.show()
    # selection = np.where(groundtruth == 0)
    Y = subsampledGT
    Y_pred = subsampledPred
    g_uniq = np.unique(Y)
    y_uniq = np.unique(Y)
    y_pred_unique = np.unique(Y_pred)
    cm = confusion_matrix(Y.tolist(), Y_pred.tolist(), labels=[0,3])
    cmtot=cm+cmtot
    cmtotnorm = cmtot.astype('float') / cmtot.sum(axis=1)[:, np.newaxis]
    EvaluationAccuracy = np.sum(np.diag(cmtotnorm)) / np.sum(cmtotnorm)
    EvaluationSpecificity = cmtotnorm[1, 1] / np.sum(cmtotnorm[:, 1])
    EvaluationSensitivity = cmtotnorm[0, 0] / np.sum(cmtotnorm[:, 0])


    print(cm)
    print(cmtotnorm)
    print(EvaluationAccuracy)
    print(EvaluationSpecificity)
    print(EvaluationSensitivity)



    #exit(2)
    # cm = np.asarray(cm).astype(int)
    # cmnorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cmanalysis.append(cmnorm)
    # cmtot = cmtot + cm

    # accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    # print(accuracy)

    # y_pred = cross_val_predict(clf, X, Y, cv=10)
    # cm = np.asarray(confusion_matrix(Y, y_pred))
    # accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # p_class.extend(prediction.tolist())
    # t_class.extend(Y_test)

print(cmtot)
cmtotnorm = cmtot.astype('float') / cmtot.sum(axis=1)[:, np.newaxis]
EvaluationAccuracy = np.sum(np.diag(cmtotnorm)) / np.sum(cmtotnorm)
EvaluationSpecificity = cmtotnorm[1, 1] / np.sum(cmtotnorm[:, 1])
EvaluationSensitivity = cmtotnorm[0, 0] / np.sum(cmtotnorm[:, 0])
print(cmtotnorm)
print('Complete')
exit(4)

# cm = np.asarray(confusion_matrix(t_class, p_class))
# print(cm)
# accuracy = np.sum(cm.diagonal()) / np.sum(cm)
# print(accuracy)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm)
# exit(0)
