import json
import random
from keras.utils import to_categorical
# from tensorflow.keras.layers.advanced_activations import ELU
# import keras.backend.tensorflow_backend as ktf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from config import *
from dataread import fetchAnnotatedFilenames,fetchDataAndLabelsCNN,fetchDataAndLabelsML
from network import build_model
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

filenames,annotation=fetchAnnotatedFilenames()
feature="spect"

X, Y = fetchDataAndLabelsML(filenames, annotation, classes, feature,loadAnnotatedSubsetOnly=True)


countDrug = Y.count(0)
countExhale = Y.count(1)
countInhale= Y.count(2)
countNoise = Y.count(3)

print("OK")

