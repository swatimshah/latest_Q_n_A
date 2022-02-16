from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import tensorflow
from numpy import mean
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale

# load the test data
X = loadtxt('d:\\a_eeg\\shuffled_test.csv', delimiter=',')

input = X[:, 0:640]

# transform the data in the format which the model wants
input = input.reshape(len(input), 10, 64)
input = input.transpose(0, 2, 1)

# get the expected outcome 
y_real = X[:, -1]

# load the model
model = load_model('D:\\a_eeg\\model_conv1d.h5')

# get the "predicted class" outcome
y_pred = model.predict(input) 
print(y_pred.shape)
y_max = numpy.argmax(y_pred, axis=1)

# calculate the confusion matrix
matrix = confusion_matrix(y_real, y_max)
print(matrix)
