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

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from config import *
from config import results
from dataread import fetchAnnotatedFilenames, fetchDataAndLabelsML
from sklearn.model_selection import cross_val_predict
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import json


def findBestFit(X):
    results = []
    bic = []
    for components in range(1, 32):
        for cov in ['full', 'diag']:
            for tolerance in [1e-3, 1e-5, 1e-8]:
                GM = GaussianMixture(n_components=components,
                                     covariance_type=cov,
                                     max_iter=2000,
                                     tol=tolerance,
                                     random_state=0).fit(X)
                results.append(GM)
                bic.append(GM.bic(X))
    bicmin = np.argmin(np.asarray(bic))
    return results[bicmin]




np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


resultsTarget='../results/json_data.json'
print("Evaluation with 10-fold cross validation ... please be patient.")
for classifierSelect in ["rf"]:
    for feature in ["spect"]:
        for testscheme in ["multi"]:
            for loadAnnotatedSubsetOnly in [True]:
                if not loadAnnotatedSubsetOnly:
                    mixing = "mixed"
                if loadAnnotatedSubsetOnly:
                    mixing = "ummixed"

                print(classifierSelect + " " + feature + " " + testscheme + " " + " " + mixing)

                fromScratch = True
                doStore = False
                p_class = []
                t_class = []

                if testscheme == "multia0":
                    filenames, annotation = fetchAnnotatedFilenames(['a0'])
                    random.Random(9).shuffle(filenames)
                    kf = KFold(n_splits=numberOfFolds, shuffle=True)
                    splitfmns = kf.split(filenames)

                if testscheme == "multi":
                    filenames, annotation = fetchAnnotatedFilenames()
                    random.Random(9).shuffle(filenames)
                    kf = KFold(n_splits=numberOfFolds, shuffle=True)
                    splitfmns = kf.split(filenames)

                if testscheme == "loso":
                    filenames, annotation = fetchAnnotatedFilenames()
                    splitfmns = []
                    train_files, _ = fetchAnnotatedFilenames(['f1', 'g1'])
                    test_files, _ = fetchAnnotatedFilenames(['a1'])
                    train_fnms = [i for i, e in enumerate(filenames) if e in train_files]
                    test_fnms = [i for i, e in enumerate(filenames) if e in test_files]
                    splitfmns.append((train_fnms, test_fnms))
                    train_files, _ = fetchAnnotatedFilenames(['f1', 'a1'])
                    test_files, _ = fetchAnnotatedFilenames(['g1'])
                    train_fnms = [i for i, e in enumerate(filenames) if e in train_files]
                    test_fnms = [i for i, e in enumerate(filenames) if e in test_files]
                    splitfmns.append((train_fnms, test_fnms))
                 

                if testscheme == "single":
                    splitfmns = []
                    _, annotation = fetchAnnotatedFilenames()
                    filenames, _ = fetchAnnotatedFilenames(['f1'])
                    random.Random(9).shuffle(filenames)
                    kf = KFold(n_splits=numberOfFolds, shuffle=True)
                    splitfmns = kf.split(filenames)
            
                currentFold=0
                for train_fnms, test_fnms in splitfmns:
                    currentFold=currentFold+1
                    print("Training verbose is inactive... please be patient")

                    train_files = [filenames[i] for i in train_fnms]
                    X, Y = fetchDataAndLabelsML(train_files, annotation, classes, feature,loadAnnotatedSubsetOnly=True)
                    test_files = [filenames[i] for i in test_fnms]
                    X_test, Y_test = fetchDataAndLabelsML(test_files, annotation, classes, feature,
                                                          loadAnnotatedSubsetOnly=loadAnnotatedSubsetOnly)

                    data_test = np.asarray(X_test)
                    data_train = np.asarray(X)
                    label_train = np.asarray(Y)
                    label_test = np.asarray(Y_test)

                    if classifierSelect == 'ada':
                        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=600,
                                                 learning_rate=1)
                        clf.fit(X, Y)
                        prediction = clf.predict(data_test)

                    if classifierSelect == 'svm':
                        clf = make_pipeline(StandardScaler(), svm.SVC(decision_function_shape='ovr'))
                        # clf=svm.SVC(tol=1e-9)
                        clf.fit(X, Y)
                        prediction = clf.predict(data_test)

                    if classifierSelect == 'rf':
                        clf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=300)
                        clf.fit(X, Y)
                        prediction = clf.predict(data_test)

                    if classifierSelect == 'gmm':
                        F_Dr = data_train[label_train == 0, :]
                        F_Ex = data_train[label_train == 1, :]
                        F_Inh = data_train[label_train == 2, :]
                        F_Noise = data_train[label_train == 3, :]

                        GM_Dr = findBestFit(F_Dr)
                        GM_Ex = findBestFit(F_Ex)
                        GM_Inh = findBestFit(F_Inh)
                        GM_Noise = findBestFit(F_Noise)

                        pGMDR = GM_Dr.score_samples(data_test)
                        pGMEX = GM_Ex.score_samples(data_test)
                        pGMINH = GM_Inh.score_samples(data_test)
                        pGMNOISE = GM_Noise.score_samples(data_test)

                        predictionScore = np.concatenate((
                            np.expand_dims(pGMDR, axis=1),
                            np.expand_dims(pGMEX, axis=1),
                            np.expand_dims(pGMINH, axis=1),
                            np.expand_dims(pGMNOISE, axis=1)), axis=1)
                        prediction = np.argmax(predictionScore, axis=1)

                 

                    p_class.extend(prediction.tolist())
                    t_class.extend(Y_test)
                    cm = np.asarray(confusion_matrix(t_class, p_class, labels=[0, 1, 2, 3]))
                    
                    print("Confusion matrix (non-normalized) for fold : #"+ str (currentFold))
                    print("Current method evaluated:" + classifierSelect + " " + feature + " " + testscheme + " " + " " + mixing)
                    print(cm)

                 
                cm = np.asarray(confusion_matrix(t_class, p_class, labels=[0, 1, 2, 3]))
                cmnorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                accuracy = np.sum(cm.diagonal()) / np.sum(cm)
                
                # Open results json file
                with open(resultsTarget) as json_file:
                    data = json.load(json_file)
                data[classifierSelect][feature][testscheme][mixing]['accuracy'] = accuracy
                data[classifierSelect][feature][testscheme][mixing]['cm'] = cm.tolist()
                data[classifierSelect][feature][testscheme][mixing]['cmnorm'] = cmnorm.tolist()
                
                # Write to results
          
                print("Detailed outcome in json_data.json")
                print("Accuracy is:")
                print(data[classifierSelect][feature][testscheme][mixing]["accuracy"])
                print("Confusion matrix is:")
                print(data[classifierSelect][feature][testscheme][mixing]["cm"])
                print("Normalized confusion matrix is:")
                print(data[classifierSelect][feature][testscheme][mixing]["cmnorm"])

                # Write to file
                with open(resultsTarget, 'w') as outfile:
                    json.dump(data, outfile)

                
                