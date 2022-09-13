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

"""
This code evaluates the LSTM based classification of PMDI inhale audio sounds
"""


from keras.utils import to_categorical
# from tensorflow.keras.layers.advanced_activations import ELU
# import keras.backend.tensorflow_backend as ktf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
import json
from config import *
from dataread import fetchAnnotatedFilenames,fetchDataAndLabelsML,fetchDataAndLabelsCNN,fetchDataAndLabelsLSTM
from network import build_model
import itertools
from lstm.config import LstmConfig
from lstm.modeling import inhalerPredictor
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

resultsTarget='../results/json_data.json'
print("Evaluation with 10-fold cross validation ... please be patient.")

for classifierSelect in ["lstm"]:
    for feature in ["spect"]:
        for testscheme in ["multi", "single", "loso"]:
            for loadAnnotatedSubsetOnly in [True,False]:
                if not loadAnnotatedSubsetOnly:
                    mixing = "mixed"
                if loadAnnotatedSubsetOnly:
                    mixing = "ummixed"
                p_class = []
                t_class = []
                fromScratch = True

                lconf = LstmConfig()
                

                doStore = False
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

                
                # generatorToTest=splitfmns
                generatorToTest = itertools.islice(splitfmns, nFoldToRun)
                for train_fnms, test_fnms in generatorToTest:
                    # Trainset
                    currentFold=currentFold+1
                    print("Training ... please be patient")

                    predictor = inhalerPredictor(hparams=lconf)

                    train_files = [filenames[i] for i in train_fnms]
                    X, Y = fetchDataAndLabelsLSTM(train_files, annotation, classes, feature,loadAnnotatedSubsetOnly=True)
                    label_train = to_categorical(Y, num_classes=4)

                    # Testset
                    test_files = [filenames[i] for i in test_fnms]
                    X_test, Y_test = fetchDataAndLabelsLSTM(test_files, annotation, classes, feature,
                                                            loadAnnotatedSubsetOnly=loadAnnotatedSubsetOnly)
                    label_test = to_categorical(Y_test, num_classes=4)

                    

                    predictor.model.fit(X, label_train, batch_size=64, epochs=25, verbose=0)

                    predictions = predictor.model.predict(X_test)

                    t_class.extend(np.argmax(label_test, axis=1).tolist())
                    p_class.extend(np.argmax(predictions, axis=1).tolist())
                    cm = confusion_matrix(t_class, p_class)
                    print("Confusion matrix (non-normalized) for fold : #"+ str (currentFold))
                    print("Current method evaluated:" + classifierSelect + " " + feature + " " + testscheme + " " + " " + mixing)
                    print(cm)

                cm = np.asarray(confusion_matrix(t_class, p_class, labels=[0, 1, 2, 3]))
                cmnorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                accuracy = np.sum(cm.diagonal()) / np.sum(cm)
                
                # Open results json file
                with open(resultsTarget,'r') as json_file:
                    data = json.load(json_file)
                    data[classifierSelect][feature][testscheme][mixing]['accuracy'] = accuracy
                    data[classifierSelect][feature][testscheme][mixing]['cm'] = cm.tolist()
                    data[classifierSelect][feature][testscheme][mixing]['cmnorm'] = cmnorm.tolist()

                with open(resultsTarget, 'w') as outfile:
                    json.dump(data, outfile)
                
                # Write to results
          
                print("Detailed outcome in json_data.json")
                print("Accuracy is:")
                print(data[classifierSelect][feature][testscheme][mixing]["accuracy"])
                print("Confusion matrix is:")
                print(data[classifierSelect][feature][testscheme][mixing]["cm"])
                print("Normalized confusion matrix is:")
                print(data[classifierSelect][feature][testscheme][mixing]["cmnorm"])

              