# Revisiting Content Based Audio Classification for Asthma Medication Adherence



## Description

Asthma is a common, usually long-term respiratory disease with negative impact on society and the economy worldwide. Treatment involves using medical devices (inhalers) that distribute medicationto the airways, and its efficiency depends on the precision of the inhalation technique. Health monitoring systems equipped with sensors and embedded with sound signal detection enable the recognition of drug actuation and could be powerful tools for reliable audio content analysis. This repository includes a set of tools for audio processing, feature extraction and classification and is provided along with a dataset consisting of respiratory and drug actuation sounds. The classification models are implemented based on machine learning and deep approaches. This study provides a comparative evaluation of the implemented approaches, examines potential improvements and discusses challenges and future tendencies.

## Demo on code ocean

https://codeocean.com/capsule/2421402/tree

## Folder structure

```
├── code
│   ├── lstm
│    
│ 
├── data
│   ├── _a0
│   ├── _a1
│   ├── _f1
│   └── _g1
├── environment
├── metadata
└── trained_models
```

## Dataset

Dataset is available in the following links:

- [Code ocean](https://codeocean.com/capsule/2421402/tree)
- [IEEE Dataport](https://ieee-dataport.org/documents/respiratory-and-drug-actuation-dataset)

## To run on your own environment

### Requirements

- Python 3.8 (tested)
- Keras
- Tensorflow
- Pandas
- Scikit-learn
- Matplotlib

### Running the code


#### Evaluate CNN

```
cd code
python evaluate_cnn.py
```



#### Evaluate LSTM


```
cd code
python evaluate_lstm.py
```




#### Evaluate ML


```
cd code
python evaluate_ml.py
```

