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
This code evaluates the CNN based classification of PMDI inhale audio sounds
"""


import json
import random
from keras.utils import to_categorical
# from tensorflow.keras.layers.advanced_activations import ELU
# import keras.backend.tensorflow_backend as ktf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from config import *
import itertools
from dataread import fetchAnnotatedFilenames,fetchDataAndLabelsCNN
from network import build_model
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

resultsTarget='../results/json_data.json'
print("Evaluation with 10-fold cross validation ... please be patient.")
fromScratch=True
doStore=False
for classifierSelect in ["cnn"]:
    for feature in ["time"]:
        for testscheme in ["multi", "single", "loso"]:
            for loadAnnotatedSubsetOnly in [True,False]:


                if not loadAnnotatedSubsetOnly:
                    mixing="mixed"
                if loadAnnotatedSubsetOnly:
                    mixing = "ummixed"


                p_class = []
                t_class = []



                if testscheme=="multia0":
                    filenames,annotation=fetchAnnotatedFilenames(['a0'])
                    random.Random(9).shuffle(filenames)
                    kf = KFold(n_splits=numberOfFolds, shuffle=True)
                    splitfmns=kf.split(filenames)


                if testscheme=="multi":
                    filenames,annotation=fetchAnnotatedFilenames()
                    random.Random(9).shuffle(filenames)
                    kf = KFold(n_splits=numberOfFolds, shuffle=True)
                    splitfmns=kf.split(filenames)


                if testscheme=="loso":
                    filenames,annotation=fetchAnnotatedFilenames()
                    splitfmns=[]
                    train_files,_=fetchAnnotatedFilenames(['f1','g1'])
                    test_files,_=fetchAnnotatedFilenames(['a1'])
                    train_fnms=[i for i, e in enumerate(filenames) if e in train_files]
                    test_fnms=[i for i, e in enumerate(filenames) if e in test_files]
                    splitfmns.append((train_fnms,test_fnms))
                    train_files,_=fetchAnnotatedFilenames(['f1','a1'])
                    test_files,_=fetchAnnotatedFilenames(['g1'])
                    train_fnms=[i for i, e in enumerate(filenames) if e in train_files]
                    test_fnms=[i for i, e in enumerate(filenames) if e in test_files]
                    splitfmns.append((train_fnms,test_fnms))



                if testscheme=="single":
                    splitfmns=[]
                    _,annotation=fetchAnnotatedFilenames()
                    filenames,_=fetchAnnotatedFilenames(['f1'])
                    random.Random(9).shuffle(filenames)
                    kf = KFold(n_splits=numberOfFolds, shuffle=True)
                    splitfmns=kf.split(filenames)


                currentFold=0   
                # generatorToTest=splitfmns
                generatorToTest = itertools.islice(splitfmns, nFoldToRun)
                for train_fnms, test_fnms in generatorToTest:
                    currentFold=currentFold+1
                    print("Training verbose is inactive... please be patient")
                    # Trainset
                    train_files = [filenames[i] for i in train_fnms]
                    X,Y=fetchDataAndLabelsCNN(train_files,annotation,classes,loadAnnotatedSubsetOnly=True)
                    data_train=np.asarray(X)
                    label_train=to_categorical(Y)
                    data_train.resize(len(data_train),250,16,1)

                    # Testset
                    test_files = [filenames[i] for i in test_fnms]
                    X_test,Y_test=fetchDataAndLabelsCNN(test_files,annotation,classes,loadAnnotatedSubsetOnly=loadAnnotatedSubsetOnly)
                    data_test=np.asarray(X_test)
                    label_test=to_categorical(Y_test)
                    data_test.resize(len(data_test),250,16,1)



                    inhaler_model = build_model(data_train.shape[1:],num_classes)
                    inhaler_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                    # inhaler_model.summary()
                    if fromScratch:
                        inhaler_train = inhaler_model.fit(data_train, label_train,batch_size=batch_size,epochs=epochs,verbose=2)
                        if doStore:
                            inhaler_model.save(trainedModel)
                    else:
                        inhaler_model.load_weights(trainedModel)


                    prediction = inhaler_model.predict(data_test)
                    p_class.extend(np.argmax(prediction, axis=1).tolist())
                    t_class.extend(np.argmax(label_test, axis=1).tolist())
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
          
                # print(json.dumps(data[classifierSelect][feature][testscheme][mixing], indent=4, sort_keys=True))
                print("Detailed outcome in json_data.json")
                print("Accuracy is:")
                print(data[classifierSelect][feature][testscheme][mixing]["accuracy"])
                print("Confusion matrix is:")
                print(data[classifierSelect][feature][testscheme][mixing]["cm"])
                print("Normalized confusion matrix is:")
                print(data[classifierSelect][feature][testscheme][mixing]["cmnorm"])



