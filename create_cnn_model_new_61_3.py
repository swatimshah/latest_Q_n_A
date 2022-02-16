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
from keras import optimizers

# setting the seed
seed(1)
set_seed(1)

# load training data
X_train_whole = loadtxt('d:\\a_eeg\\EEGData_1024.csv', delimiter=',')

# load the test data
X = loadtxt('d:\\a_eeg\\EEGData_1024_Test.csv', delimiter=',')

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
savetxt('d:\\a_eeg\\shuffled_test.csv', data_combined[index1:index1+index2, :], delimiter=',') 

# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(data_combined[0:index1, :], data_combined[0:index1, -1], random_state=1, test_size=0.3, shuffle = True)
print(X_train_tmp.shape)
print(X_test_tmp.shape)


# augment train data
choice = X_train_tmp[:, -1] == 0.
X_total_1 = numpy.append(X_train_tmp, X_train_tmp[choice, :], axis=0)
X_total_2 = numpy.append(X_total_1, X_train_tmp[choice, :], axis=0)
X_total_3 = numpy.append(X_total_2, X_train_tmp[choice, :], axis=0)
X_total_4 = numpy.append(X_total_3, X_train_tmp[choice, :], axis=0)
X_total = numpy.append(X_total_4, X_train_tmp[choice, :], axis=0)


print(X_total.shape)

# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total, X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))


#=======================================
 
# Data Pre-processing - presently nothing

input = X_train_keep[:, 0:640]
testinput = X_test_tmp[:,0:640]
Y_train = Y_train_keep
Y_test = Y_test_tmp

#=====================================

# Model configuration

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 10, 64)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 10, 64)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)



# Create the model
model=Sequential()
model.add(Conv1D(filters=60, kernel_size=5, padding='valid', activation='relu', strides=1, input_shape=(64, 10)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=60, kernel_size=5, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.4))
model.add(GlobalAveragePooling1D())
model.add(Dense(2, activation='softmax'))

model.summary()

# Compile the model
sgd = optimizers.SGD(lr=0.01, momentum=0.7, nesterov=True)       
model.compile(loss=sparse_categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

hist = model.fit(input, Y_train, batch_size=24, epochs=300, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None)


# evaluate the model
Y_hat_classes = model.predict_classes(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.xlabel("No of iterations")
pyplot.ylabel("Accuracy and loss")
pyplot.show()

#==================================

model.save("D:\\a_eeg\\model_conv1d.h5")

#==================================

#Removed dropout and reduced momentum and reduced learning rate