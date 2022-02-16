from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import tensorflow
from numpy import mean
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale
import pickle


# load the test data
X = loadtxt('d:\\b_eeg\\shuffled_test.csv', delimiter=',')


# Data Pre-processing - Extract the first PCA component
input = X[:, 0:64]

# get the expected outcome 
y_real = X[:, -1]
print(y_real)

# load the model
model_file = open('svm_model.sav', 'rb')
model = pickle.load(model_file)

# get the "predicted class" outcome
y_pred = model.predict(input) 
print(y_pred.shape)
print(y_pred)

# calculate the confusion matrix
matrix = confusion_matrix(y_real, y_pred)
print(matrix)
