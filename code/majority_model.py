'''
Implementation for the majority model used as baseline
'''
import os 
import numpy as np
import numpy.random as npr
import argparse
import pickle
from sklearn.utils import shuffle
import util
from sklearn.utils import check_random_state

def countFrequencies(dataY, save="yes", saveFile="work/frequencies.pickle"):
	'''
	Count the frequencies of the data

	:param dataY: the output data
	:param save: if "yes" save the data in the specified file
	:param saveFile: where to store the frequencies

	:return freqs: the frequencies
	'''
	freqs = {}
	for sample in dataY:
		for i in range(dataY.shape[1]):
			if sample[i] > 0:
				if i not in freqs.keys():
					freqs[i] = 1
				else:
					freqs[i] += 1

	if save == "yes":
		with open("unnormalized.pickle", "wb") as handler:
			pickle.dump(freqs, handler)

	for k in range(dataY.shape[1]):
		freqs[k] = freqs[k] / dataY.shape[0]

	if save == "yes":
		with open(saveFile, 'wb') as handler:
			pickle.dump(freqs, handler)
		
	return freqs



def createModel(frequencies, threshold=0.5):
	'''
	Generate the model

	:param frequencies: the frequencies previously calculated or loaded
	:param threshold: the threshold to use for the classification

	:return model: the majority model
	'''
	model = {}
	for freq in frequencies:
		if frequencies[freq] > threshold:
			model[freq] = 1
		else:
			model[freq] = 0
	return model


def predict(dataX, dataY, model):
	'''
	Predict ingredients for the nutrients given.

	:param dataX: the input data
	:param dataY: the input target data
	:param model: the model to be used for the predictions

	:return the predictions
	'''
	predictions = np.zeros(shape=(dataY.shape[0],dataY.shape[1]))
	for i in range(dataX.shape[0]):
		for j in model:
			if model[j] == 1:
				predictions[i][j] = 1
			else:
				predictions[i][j] = 0

	return predictions
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='../data/data.also.original.pickle.reverted')
	parser.add_argument('--load_frequencies', type=str, default="yes")
	parser.add_argument('--file_frequencies', type=str, default="work/frequencies.pickle")
	parser.add_argument("--save_frequencies", type=str, default="yes")
	parser.add_argument('--file_accuracies', type=str, default="work/accuracies_baseline.results")
	args = parser.parse_args()
	if args.data:
		print("--Loading data from:" + args.data)
		with open(args.data, 'rb') as f:
			data = pickle.load(f)
			(dataX, dataY) = data["xs"], data["ys"]


	xs_train, ys_train, xs_validation, ys_validation = util.split_dataset(dataX,dataY)

	if args.load_frequencies == "yes":
		with open(args.file_frequencies, 'rb') as handler:
			freq = pickle.load(handler)
		print("--Frequencies loaded!")
	else:
		freq = countFrequencies(ys_train,args.save_frequencies, args.file_frequencies)
		print("--Frequencies counted and saved!")

	model = createModel(freq)
	print("--Model created!")

	predictions = predict(xs_validation, ys_validation, model)
	mean_Acc = util.macroAccuracy(ys_validation, predictions)
	ps,rs,fs,ss = util.macroPreRecF1(ys_validation, predictions)
	f1 = util.macroF1(ys_validation, predictions)

	#util.saveAccuracies(ys_validation, predictions, args.file_accuracies)
	util.saveMetrics(ys_validation, predictions, args.file_accuracies)
	print("--Metrics saved on the specified folder!")

if __name__=='__main__':
    main()