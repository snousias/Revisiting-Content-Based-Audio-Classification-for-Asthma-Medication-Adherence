# Evaluation of content based audio classification algorithms for asthma medication adherence 

## Introduction

Asthma is a common, usually long-term respiratory disease with negative impact on society and the economy worldwide. Treatment involves using medical devices (inhalers) that distribute medicationto the airways, and its efficiency depends on the precision of the inhalation technique. Health monitoring systems equipped with sensors and embedded with sound signal detection enable the recognition of drug actuation and could be powerful tools for reliable audio content analysis. This paper revisits audio pattern recognition  and  machine  learning  techniques  for  asthma  medication  adherence  assessment  and  presents the Respiratory and Drug Actuation (RDA)Suite  for benchmarking and further research.  This suite includes  a  set  of  tools  for  audio  processing,  feature  extraction  and  classification  and  is  provided  along with a dataset consisting of respiratory and drug actuation sounds.  This study compares a series of classifiers namely SVM, Random Forests, AdaBoost, LSTMs, CNN for spectral, cepstral, MFCC and temporal features.


## Demo on code ocean with data


https://codeocean.com/capsule/8383844/tree/


## Dataset on IEEE Dataport

https://ieee-dataport.org/documents/respiratory-and-drug-actuation-dataset

## Dataset
Generic format:
```
Filename, Class, Sample index at the beginning of the acoustic event, Sample index at the end of the acoustic event

```

Example:

```
rec2018-01-22_17h41m33.475s.wav,Exhale,6015,17437
rec2018-01-22_17h41m33.475s.wav,Inhale,20840,31655
rec2018-01-22_17h41m33.475s.wav,Drug,31898,37610
rec2018-01-22_17h41m33.475s.wav,Exhale,43686,59969
rec2018-01-22_17h41m49.809s.wav,Inhale,5043,17316
rec2018-01-22_17h41m49.809s.wav,Drug,18288,24364
rec2018-01-22_17h41m49.809s.wav,Exhale,31412,46724
rec2018-01-22_17h42m07.718s.wav,Exhale,303,9782
rec2018-01-22_17h42m07.718s.wav,Inhale,16951,28010
```

## Configuration

Set the proper location for trained model and data path in config.py

```
batch_size = 32
epochs = 25
classes=['Drug','Exhale','Inhale','Noise']
trainedModel = './trained_models/inhaler_model_5.h5py'
fromScratch=False
doStore=False
path="data/"
```

## Evaluation

### Training and testing machine learning models with 10-fold cross validation
```

python evaluate_ml_models.py

```

### Training and testing LSTM models with 10-fold cross validation

```

python evaluate_lstm.py

```



### Training and testing CNN models with 10-fold cross validation

```

python evaluate_cnn.py

```



