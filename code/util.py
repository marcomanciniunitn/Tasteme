'''
Script containing util functionalities used both from the ICNN model and from the Majority model.
'''
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.utils import check_random_state
import csv


#np.set_printoptions(threshold=np.nan)
def macroF1(trueY, predY):
    '''
	Function used to compute the global F1 

	:param trueY: the true labels
	:param predY: the predicted labels

	:return f1_score: the global F1 Score
	'''

    predY_bin = (predY >= 0.5).astype(np.int)
    trueY_bin = trueY.astype(np.int)

    # The transpose is here because sklearn's f1_score expects multi-label
    # data to be formatted as (nLabels, nExamples).
    return f1_score(trueY_bin.T, predY_bin.T, average='macro', pos_label=None)

def macroPreRecF1(trueY, predY):
	'''
	Function returning the  precisions, recall, support and f1 scores on the single classes.

	:param trueY: the true labels
	:param predY: the predicted labels
	'''
	predY_bin = (predY >= 0.5).astype(np.int)
	trueY_bin = trueY.astype(np.int)

	return precision_recall_fscore_support(trueY_bin, predY_bin, average=None)


def macroAccuracy(trueY, predY):
	'''
	Used to calculate the global accuracy, calculated as mean of the accuracies on single elements

	:param trueY: the true labels.
	:param predY: the predicted labels.
	'''
	predY_bin = (predY >= 0.5).astype(np.int)
	trueY_bin = trueY.astype(np.int)

	accuracies = []
	for i in range(predY.shape[1]):
		single_accuracy = accuracy_score(trueY_bin[:, i], predY_bin[:, i])
		accuracies.append(single_accuracy)
	
	mean_acc = np.mean(accuracies)
	return (mean_acc, accuracies)

def saveAccuracies(trueY, predY, toSave):
	'''
	Calculate and save accuracies on single classes.

	:param trueY: the true labels.
	:param predY: the predicted labels.
	:param toSave: file where to store the accuracies
	'''
	predY_bin = (predY >= 0.5).astype(np.int)
	trueY_bin = trueY.astype(np.int)
	accuracies = []
	for i in range(predY.shape[1]):
		single_accuracy = accuracy_score(trueY_bin[:, i], predY_bin[:, i])
		accuracies.append(single_accuracy)

	np.savetxt(toSave, accuracies, delimiter=",")

def saveMetrics(trueY, predY, toSave):
	'''
	Function used to calculate and save precision, recall, f1 and accuracies over single classes.

	:param trueY: the true labels
	:param predY: the predicted labels.
	:param toSave: the file to use as storage for the metrics
	'''
	predY_bin = (predY >= 0.5).astype(np.int)
	trueY_bin = trueY.astype(np.int)
	accuracies = []
	ps, rc, f1, ss = precision_recall_fscore_support(trueY_bin, predY_bin, average=None)
	for i in range(predY.shape[1]):
		single_accuracy = accuracy_score(trueY_bin[:, i], predY_bin[:, i])
		metrics = (float(single_accuracy), float(ps[i]), float(rc[i]), float(f1[i]))
		accuracies.append(metrics)

	with open(toSave,'w') as out:
		csv_out=csv.writer(out)
		csv_out.writerow(["accuracy", "precision", "recall", "f1"])
		for row in accuracies:
			csv_out.writerow(row)

def macroSquaredError(trueY, predY):
	'''
	Function used to compute the global MSE.

	:param trueY: the true labels.
	:param predY: the predicted labels.

	:return mean_squared_error: the global MSE.
	'''

	trueY = trueY.astype(float)
	predY = predY.astype(float)
	return mean_squared_error(trueY, predY)

def split_dataset(xs, ys, faction_valid=0.2, seed=0):
    """
    @Author: Pedranza
    Split the dataset into train and validation.
    :param xs: Input matrix.
    :param ys: Output matrix.
    :param faction_valid: Fraction of the training example to use for the validation.
    :param seed: Seed used for the random split.
    :return: X_train, y_train, X_validation, y_validation.
    """

    # check the size
    num_examples = xs.shape[0]
    assert num_examples == ys.shape[0]

    # fix the random generator
    rng = check_random_state(seed)

    # random split
    num_valid_examples = int(np.rint(num_examples * faction_valid))
    indexes = rng.permutation(num_examples)

    # train
    train_indices = indexes[num_valid_examples:]
    train_xs, train_ys = xs[train_indices], ys[train_indices]

    # validation
    valid_indices = indexes[:num_valid_examples]
    valid_xs, valid_ys = xs[valid_indices], ys[valid_indices]

    # return the split
    return train_xs, train_ys, valid_xs, valid_ys