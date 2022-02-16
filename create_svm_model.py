import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
import tensorflow
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
import pickle
from sklearn.naive_bayes import GaussianNB


# setting the seed
seed(1)
set_seed(1)

# load training data
X_train_whole = loadtxt('d:\\b_eeg\\EEGData_1024.csv', delimiter=',')

# load the test data
X = loadtxt('d:\\b_eeg\\EEGData_1024_Test.csv', delimiter=',')

# shuffle the train data
numpy.random.shuffle(X_train_whole)

# shuffle the test data
numpy.random.shuffle(X)

# combine the train and test data
data_combined = numpy.append(X_train_whole, X, axis=0)

# shuffle the combined data
numpy.random.shuffle(data_combined)
print(data_combined.shape)

index1 = len(X_train_whole)
index2 = len(X)

# Divide 398 shuffled samples for training and 398 shuffled samples for testing. This will mix the readings taken in 2 different batches. 
savetxt('d:\\b_eeg\\shuffled_test.csv', data_combined[index1:index1+index2, :], delimiter=',') 

# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(data_combined[0:index1, :], data_combined[0:index1, -1], random_state=1, test_size=0.3, shuffle = True)
print(X_train_tmp.shape)
print(X_test_tmp.shape)




#=======================================
 
# Data Pre-processing - Extract the first PCA component

input = X_train_tmp[:, 0:64]
testinput = X_test_tmp[:, 0:64]
Y_train = Y_train_tmp
Y_test = Y_test_tmp

print(Y_train)
print(Y_test)
#=====================================

# Model configuration

print (input.shape)
print (testinput.shape)

# Create SVM model

model = SVC(C=22, kernel='poly', verbose=True, coef0=0.2, degree=4, gamma='scale')  
model.fit(input, Y_train)

# evaluate the model
Y_hat_classes = model.predict(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)

#==================================

model_file = open('d:\\b_eeg\\svm_model.sav', 'wb')
pickle.dump(model, model_file)

#==================================

#Removed dropout and reduced momentum and reduced learning rate