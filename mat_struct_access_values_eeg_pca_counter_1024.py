import scipy.io as sio
import numpy
from numpy import savetxt 
from sklearn.decomposition import PCA
from numpy.random import seed
from tensorflow.random import set_seed


# setting the seed
seed(1)
set_seed(1)

def _check_keys( dict):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict:
    		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
        		dict[key] = _todict(dict[key])
	return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
	"""
	this function should be called instead of direct scipy.io .loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	"""
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)

combinedData = numpy.empty([0, 640])

myKeys = loadmat("d:\\b_eeg\\EEGData_1024.mat")
print(myKeys)
eegData = myKeys['EEGData']
eegDataAllSamples = eegData['Data']
eegDataAllLabels = eegData['Labels']
print(eegDataAllSamples.shape)
print(eegDataAllLabels.shape)

yes_input = eegDataAllLabels == 'Yes'
eegDataAllLabels[yes_input] = 1
no_input = eegDataAllLabels == 'No'
eegDataAllLabels[no_input] = 0


for i in range (398):
	eegData = eegDataAllSamples[i].reshape(64, 1024)
	eegData = eegData.transpose()
	pca_target = PCA(n_components=10, random_state=2)
	pca_target.fit(eegData)
	print(pca_target.components_.shape)
	combinedData = numpy.append(combinedData, pca_target.components_.flatten().reshape(1, 640), axis=0)

wholeData = numpy.append(combinedData, eegDataAllLabels.reshape(len(eegDataAllLabels), 1), axis=1)

savetxt('d:\\b_eeg\\EEGData_1024.csv', wholeData, delimiter=',')


