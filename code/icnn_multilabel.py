#!/usr/bin/env python3
'''
This is the main script of the Tasteme project aiming at suggesting new products to the end user. 
'''
import tensorflow as tf
import tflearn

import numpy as np
import numpy.random as npr

np.set_printoptions(precision=2)
np.seterr(all='raise')

import argparse
import csv
import os
import sys
import time
import pickle
import json
import shutil


from datetime import datetime

import matplotlib as mpl
from matplotlib import cm
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from sklearn.utils import shuffle
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.utils import check_random_state
import util
import sqlite3
from network_classification import NeuralNetworkClassification

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def main():
    parser = argparse.ArgumentParser()
    ### MODEL PARAMETERS ###
    parser.add_argument('--save', type=str, default='work',  help='Save directory where to store the results')
    parser.add_argument('--nEpoch', type=int, default=100,  help= "Number of epochs to perform during the training")
    parser.add_argument('--testEpoch', type=int, default=20, help="Set to N means that at Nth epoch starts a testing on the validation set")
    parser.add_argument('--trainBatchSz', type=int, default=250, help="Batch size for the training phase")
    parser.add_argument('--seed', type=int, default=42, help="Seed for the shufflings")
    parser.add_argument('--model', type=str, default="picnn",
                        choices=['picnn', 'ficnn'], help="Model to choose, [picnn, ficnn]")
    parser.add_argument('--noncvx', action='store_true', default=False, help="If set remove the convexity operations on the weights")
    parser.add_argument('--layers', type=int, nargs="+", default=[200], help="Layers to put the ICNN")
    parser.add_argument("--batch_train", type=str, default="yes", help="If set to 'yes' the training becomes batched")
    parser.add_argument("--approx_iters", type=int, default=30, help="Number of iterations used to approximate the inference in the inference phase")
    parser.add_argument("--save_model", type=str, default="no", help="If set to 'yes' save the model parameters in the path specified by 'path_model'")
    parser.add_argument("--path_model", type=str, help="If 'save_model' parameter has been set to 'yes' then this path is used to save the model, otherwise this is used to load the model. In case neither this parameter nor the 'save_model' are set, then the model is trained but not saved or loaded.")

    ### DATA PARAMETERS ###
    parser.add_argument('--data', type=str, help="Pickle file containing the dataset")
    parser.add_argument("--binarize", type=str, default="no", help="If dataset is passed it is possible to binarize the output, usefull to replicate Pedranza's experiments")
    parser.add_argument("--multilabel", type=str, default="yes", help="If no dataset has been passed then it is possible to generate multilabel synthetic dataset to train and evaluate the model")
    parser.add_argument("--n_samples", type=int, default=175191, help="Number of samples to generate in a synthetic case")
    parser.add_argument("--n_features", type=int, default=513, help="Number of features to generate in a synthetic case")
    parser.add_argument("--n_classes", type=int, default=38, help="Number of classes/regression values to predict for a synthetic case")
    
    ### SAVE AND PLOT PARAMETERS ###
    parser.add_argument("--plot_acc" , type=str, default="no", help="If set to 'yes' it plots the accuracies epoch by epoch both for training and testing")
    parser.add_argument("--verbose", type=str, default="no", help="If set to 'yes' it save on the accuracy files more verbose resultsets")
    parser.add_argument("--save_accuracies", type=str, default="no", help="If set to 'yes' it saves the accuracies, precisions, recall and f1 of all the test epochs. The files are generated under the 'save' directory and their name starts with the 'file_accuracies' parameter, followed by the number of the epoch.")
    parser.add_argument("--file_accuracies", type=str, default="work/single_accuracies", help="If 'save_accuracies' parameter has been set then this parameter is used in order to give the main name to the metrics files.")
    parser.add_argument("--db_file", type=str, help="Path of the database file, usefull to solve nutrients, products and ingredients name")
    
    ### PREFERENCE TEST ARGUMENTS ###
    parser.add_argument("--test_dir", type=str, default="1/preference_results.json" )
    parser.add_argument("--test_focus", type=str, default="all", help="Parameter indicating if the increment/decrement of the nutrients will run on all the nutrients or part of them [all|part of]")
    parser.add_argument("--test_type", type=str, default="incremental", help="Parameter indicating the kind of changements to apply to the nutrients [increment | decrement]")
    parser.add_argument("--test_ratio", type=float, nargs="+" , default=[2], help="Ratio indicating the amount of changement to apply to the nutrients, it is also possible to specify specific rateos for specific nutrients, following the order of the test_nutrients parameter")
    parser.add_argument("--test_nutrients", type=int, nargs="+", default=[17,24,33,35,37], help="Nutrients to change during the test, use their position on the input array of the ICNN")
    parser.add_argument("--test_lambda", type=float, default=0.5, help="Lambda value for the preference inference weight impose to the distance between the ingredients")
    args = parser.parse_args()

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    #Loading data
    dataX, dataY, ingred_map, ys_orig, scaler, nutr_map, prod_map = loadData(args)

    #Creating save directory if not already in place.
    if not os.path.exists(os.path.join(os.getcwd(), args.save)):
        print("Save directory not existing! It has been just created!")
        os.makedirs(os.path.join(os.getcwd(), args.save))
   

    #Splitting the dataset in training and validation set
    xs_train, ys_train, xs_validation, ys_validation = util.split_dataset(dataX,dataY)

    #Gathering informations about the dataset
    nTrain = xs_train.shape[0]
    nTest = xs_validation.shape[0]
    nFeatures = xs_train.shape[1]
    nLabels = ys_train.shape[1]
    nXy = nFeatures + nLabels

    #Printing informations about the dataset
    print("\n\n" + "="*40)
    print("+ nTrain: {}, nValidation: {}".format(nTrain, nTest))
    print("+ nFeatures: {}, nLabels: {}".format(nFeatures, nLabels))
    print("="*40 + "\n\n")

    config = tf.ConfigProto() #log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
    	#Creating model
        model = Model(nFeatures, nLabels, sess, args.model, args.layers, args.approx_iters, args.test_lambda)

        #Load it if requested 
        if args.save_model == "no" and args.path_model:
        	model.load(args.path_model)
        else:
        	model.trainNew(args, xs_train, ys_train, xs_validation, ys_validation)
        
        #Save it if requested
        if args.save_model == "yes":
        	model.save(args.path_model)

        #Quantitative analysis
        '''
        (mse, acco, f1_glob, meta_accuracies, meta_f1, meta_pre, meta_rec, diff) = testPreferenceModel(model, xs_validation, ys_validation, args.test_focus, args.test_ratio, args.test_nutrients, scaler, args.db_file, nutr_map, args.save, args.test_dir)
        saveTestMetadata(args.test_focus, args.test_type, args.test_ratio, args.test_nutrients, args.test_lambda, acco, f1_glob, meta_accuracies, meta_f1, meta_pre, meta_rec, mse, args.save, args.test_dir, diff)
  		'''

  		#Qualitative analysis       
        prodIDList = [45215764,45041132,45055564,45040723,45134297, 45100704,45048752, 45052408, 45142045, 45032144 ,45205524, 45149871, 45136875, 45209321, 45051944, 45039981, 45054634, 45204490, 45003839, 45010359, 45078194, 45035425, 45034903, 45144693, 45052762,45058557, 45138324,45176120,45177715]
        qualitativeTest(model, prodIDList, args.test_nutrients, args.test_ratio, dataX, dataY, scaler, prod_map, args.db_file, ingred_map)


def saveTestMetadata(focus, test_type, ratios, nutrients, lambda_pref, accuracy, f1, accuracies, f1_scores, pre,rec , mse, save, subdir,differences):
	'''
	Save the test metadata under the specified directory.
	:param focus: all ingredients or part of [all|part_of]
	:param test_type: kind of tests to launch on ingredients changements [incremental, decrementa, mixed]
	:param ratios: ratios to apply to the nutrients to be changed [2x | 5x | 10x | 0x]
	:param nutrients: nutrients focused for the test
	:param lambda_pref: lambda value assigned to the new inference problem
	:param accuracy: global accuracy
	:param f1: global f1
	:param accuracies: accuracies on single nutrients
	:param f1_scores: f1 scores on single nutrients
	:param pre: precision scores on single nutrients
	:param rec: recall scores on single nutrients,
	:mse Mean squared error on predictions
	:save Main directory where to save the project
	:subdir Subdirectory where to save the projet
	'''
	meta = {"Focus ": focus, 
	"Type ": test_type,
	"Ratios ": ratios,
	"Nutrients ": nutrients, 
	"lambda_pref ": lambda_pref, 
	"Global Accuracy ": accuracy, 
	"Single accuracies " : accuracies,
	"Global F1 " : f1,
	"f1_scores " : f1_scores,
	"MSE ": mse,
	"Precisions": pre,
	"Recall":  rec,
	"Differences": differences}

	metaP = os.path.join(save, os.path.join(subdir, "preference_results.json"))
	with open(metaP, 'w') as f:
		json.dump(meta, f, indent=2)

def testPreferenceModel(model, X,Y, focus, ratios, focus_nutr, scaler, db_file, nutr_map, save, subdir):
	'''
	Function used to test the new model. The algorithm performs the following steps:
	1. Given a set of products X the "preferenceBasedSelection" modify the products according to the test, and return the set of modified 
	nutrients (modX) and the set of new predicted ingredients (yStar)
	2. The modified nutrients are binarized (xStar)
	3. From yStar (the new predicted ingredients) we predict a set of binarized nutrients (xTilde) using an MLP (the pedranza's one)
	4. Then we calculate the different performance mentrics over xStar and xTilde

	:param model: the ICNN model
	:param X: the set of products 
	:param Y: the set of ingredients to predict
	:param focus: the focus value for the test
	:param ratios: ratios to apply to the nutrients to be changed [2x | 5x | 10x | 0x]
	:param focus_nutr: nutrients focused for the test
	:param scaler: scaler used for roll-back the nutrients values to the original ones,  useful for the changements
	:param db_file: database file needed for the metrics printing
	:param nutr_map: mapping of the nutrients.
	:param save: main directory of the project
	:param subdir: subdirectory of the project.
	'''
	#Creating save directory if not already in place.
	if not os.path.exists(os.path.join(os.getcwd(), os.path.join(save, subdir))):
		print("Test directory just created!")
		os.makedirs(os.path.join(os.getcwd(), os.path.join(save, subdir)))

	yStar, modX = model.preferenceBasedSelection(X,Y,focus, ratios, focus_nutr, scaler)
	'''
	with open("additional_data_" + str(ratios[0]) + "x-" + str(focus_nutr[0]) + ".pickle", "wb") as file:
		toDump = { "xs" : yStar, "ys": modX}
		pickle.dump(toDump, file)

	print("Dump done")
	'''
	xStar = binarize(modX)
	xTilde = predictBinNutr(yStar)

	mse = util.macroSquaredError(xStar, xTilde)
	acco, accuracies = util.macroAccuracy(xTilde, xStar)
	pre,rec,f1,supp = util.macroPreRecF1(xTilde, xStar)
	f1_glob = util.macroF1(xTilde, xStar)

	print("####### PREFERENCE RESULTS ######")
	
	print("# Metrics on single nutrients #")
	meta_accuracies = printMetricsSingleNutrients("Accuracy", accuracies, db_file, nutr_map)
	meta_f1 = printMetricsSingleNutrients("F1", f1, db_file, nutr_map)
	meta_pre = printMetricsSingleNutrients("Precision", pre, db_file, nutr_map)
	meta_rec = printMetricsSingleNutrients("Recall", rec, db_file, nutr_map)
	
	differences = model.compareIngred(Y, yStar)

	print("# Global Metrics")
	print("MSE: " + str(mse))
	print("Global Accuracy: " + str(acco))
	print("Global F1: " + str(f1_glob))
	print("Mean difference:" + str(differences))

	printMetricsSingleNutrients("Accuracy", accuracies, db_file, nutr_map, 1,focus_nutr)
	printMetricsSingleNutrients("F1", f1, db_file, nutr_map, 1, focus_nutr)


	return (mse, acco, f1_glob, meta_accuracies, meta_f1, meta_pre, meta_rec, differences)


def printMetricsSingleNutrients(metric_name, metric, db_file, nutr_map, single=0, nutr=[]):
	'''
	Given a specific metric values array and the related name, this functions shows the performance values using the nutrients names.

	:param metric_name: the metric name (e.g "F1", "precision", ..)
	:param metric: the metric array containing the performance values
	:param db_file: the database file needed to query the nutrient names.
	:param nutr_map: the nutrient mappings needed to query the correct nutrient names.
	'''
	con = sqlite3.connect(db_file)
	db_handler = con.cursor()
	selNutrQuery = "SELECT nutrient_name FROM nutrients WHERE nutrient_id=?"
	metaData = {}

	for i in range(len(metric)):
		if single == 1:
			if i in nutr:
				key = metric_name + " @ " + str(i)
				print(key + ": " + str(metric[i]))
		else:
			db_handler.execute(selNutrQuery, (int(nutr_map[i]),))
			nutr_name = db_handler.fetchone()
			key = metric_name + " @ " + str(nutr_name[0])
			print(key + ": " + str(metric[i]))
			metaData[key] = str(metric[i])

	return metaData

def loadData(args):
	'''
	Load the data used to train and evaluate the ICNN. It is possible to use three different categories of datasets:
	1- Pickle dataset, where the inputs are under a dictionary called "xs" and targets under a dictionary called "ys". For the specific
	case of the nutrient->ingredient dataset all the mappings and scaler are loaded
	2- Synthetic multiclassification dataset where the parameters indicating the classes are specified by the arguments, by default it reproduce the nutr->ingredient dataset
	3- Synthetic regression dataset where the parameters specifying all the regression values, features, ... are specified by the arguments, by default it reproduces the ingredient->nutrient dataset
	
	:param args: the arguments passed as input to the script
	:return X,Y: dataset 
	:return ingred_map: mappings of the ingredients in the DB
	:return ys_orig: original values pre-scaled
	:return scaler: scaler used to rescale the nutrients
	:return nutr_map: mappings for the nutrients ID in the DB
	:return prod_map: mappings for the products ID in the DB
	'''
	ingred_map = {}
	ys_orig = {}
	scaler = {}
	nutr_map = {}
	prod_map = {}

	if args.data:
		print("Loading data from:" + args.data)
		with open(args.data, 'rb') as f:
			data = pickle.load(f)
			(dataX, dataY) = data["xs"], data["ys"]
			if len(data) > 2:
				(ingred_map, ys_orig, scaler, nutr_map, prod_map) = data["ingred_map"], data["ys_orig"], data["scaler"], data["nutrients_mapping"], data["products_map"]
			if args.binarize=="yes":
				dataY = binarize(dataY)
	else:
		if args.multilabel == "yes":
			(dataX, dataY) = make_multilabel_classification(n_samples=args.n_samples, n_features=args.n_features, n_classes=args.n_classes, n_labels=4, allow_unlabeled=False)
		else:
			informative = np.ceil(0.65*args.n_features)
			(dataX, dataY) = make_regression(n_samples=args.n_samples, n_features=args.n_features, n_informative=int(informative), n_targets=args.n_classes)


	return (dataX, dataY, ingred_map, ys_orig, scaler, nutr_map, prod_map)

def binarize(ys):
    """
    @Author: Pedranza
    Convert a matrix of continuous valued to a binary one.
    The output is 0 if the feature is below the mean, 1 if it is above.
    :param ys: Matrix, examples as rows, features as columns.
    :return: Converted matrix.
    """
    nutrient_means = ys.mean(0)
    nutrients_diffs = ys - nutrient_means
    nutrients_diffs[nutrients_diffs <= 0] = 0
    nutrients_diffs[nutrients_diffs > 0] = 1
    return nutrients_diffs.astype(int)

def variable_summaries(var, name=None):
    '''
	Variable summarizer for the training/testing phase
	'''
    if name is None:
        name = var.name
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stdev'):
            stdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stdev/' + name, stdev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def generatePrefSet(prodIDs, sampleSet, scaler, prod_map, ys):
	'''
	This function generate a preference set (set of products to be used for qualitative analysis) starting from a list of products IDs
	:param prodIDs: product IDs to be used for the preference set
	:param sampleSet: set from which the samples are taken
	:param scaler: scaler used for the normalizations
	:param prod_map: product mappings
	:param ys: true outputs for the sampleSet  
	:return testSet: the input preference set
	:return ysTest: the output preference set
	'''

	indexes_prd = []
	#Saving indexes of selected products
	for prod in prodIDs:
		indexes_prd.append(prod_map.index(int(prod)))
	

	if len(indexes_prd) != len(prodIDs):
		print("Some item has not been found!")
		exit();

	#Creating empty test set
	testSet = np.zeros((len(prodIDs), sampleSet.shape[1]))
	ysTest = np.zeros((len(prodIDs), ys.shape[1]))

	#Populating test set and eliminating them from main set
	i = 0
	for index in indexes_prd:
		
		testSet[i] = sampleSet[index]
		#scalerTest.append(scaler[index])
		ysTest[i] = ys[index]

		np.delete(sampleSet, indexes_prd, axis=0)
		np.delete(ys, indexes_prd, axis=0)
		#del scaler[index]
		i = i+1

	return testSet, ysTest

def predictBinNutr(yStar):
	'''
	Given a set of ingredients yStar this function loads a previously trained MLP (the one from pedranz) and predict the nutrients values to be under or over the mean,
	given the ingredients, then return the predictions. 
	
	:param yStar: the set of ingredients
	:return xTilde: the set of nutrients under or over the mean.
	'''
	nn = NeuralNetworkClassification(
		num_inputs= yStar.shape[1],
		num_outputs= 38,
		hidden_layers= 1,
		hidden_layer_size= 200,
		activation= "relu",
		seed=1
	)
	print("Loading MLP for binarized nutrients prediction...")
	nn.load("trained_MLP/1511424095603/weights.pickle")
	print("Predicting binarized nutrients....")
	xTilde = nn.predict(yStar)
	xTilde = binarize(xTilde)
	return xTilde

def suggestNewProduct(X, ingred):
	'''
	Function used for the qualitative analysis, since only the suggested ingredients were not that much easy to interpret for a new 
	product, I've decided to suggest to the end-user the most-similar product to what the model predicted after the changes on the nutrients.

	:param X: the sample set
	:param ingred: the product (set of ingredients) used as base for the comparisons
	:return minIndex: the index of the closest product
	:return X[minIndex]: the set of ingredients of such a closest product
	'''

	minIndex = -1
	minim = 38

	for prod in range(len(X)):
		#print(X[prod].shape)
		#print(ingred.shape)
		diff = np.count_nonzero(np.subtract(X[prod], ingred)) / len(X[prod])
		if diff < minim:
			minIndex = prod
			minim = diff

	return minIndex, X[minIndex]

def qualitativeTest(model, products, focus_nutr, focus_ratio, X,Y, scaler, prod_map,db_file, ing_map):
	'''
	This function performs a qualitative tests. It requires a set of products to be used (their DB IDs) and the changes to be applied to them.
	Then it predict a new set of ingredients and retrieve the most-similar product to such set of ingredients. Moreover it also shows the differences
	found in the two products, the starting one and the suggested one.

	:param model: the ICNN model to be used
	:param products: the set of product IDs to be used as preference set
	:param focus_nutr: the set of nutrients to change
	:param X: the entire input dataset
	:param Y: the entire output dataset
	:param scaler: the scaler used for the normalization
	:param prod_map: the product ID mappings
	:param db_file: the database file used to solve the names.
	:param ing_map: the ingredient mappings
	'''
	con = sqlite3.connect(db_file)
	db_handler = con.cursor()
	selProdQuery = "SELECT name FROM products WHERE ndbno=?"

	xsPref, ysPref  = generatePrefSet(products, X, scaler, prod_map, Y)

	print("[-] Qualitative test just started....\n")
	for product in range(len(products)):
		yStar, modX = model.preferenceBasedSelection(xsPref[product],ysPref[product],"part_of", focus_ratio, focus_nutr, scaler)
		newProdInd, ingreds = suggestNewProduct(Y, yStar)

		db_handler.execute(selProdQuery, (int(products[product]),))
		prod_name = db_handler.fetchone()
		print("[.] STARTING PRODUCT -> " + str(prod_name[0]))

		db_handler.execute(selProdQuery, (int(prod_map[newProdInd]),))
		new_prod_name = db_handler.fetchone()
		print("[.] NEW SUGGESTED PRODUCT - " + str(prod_map[newProdInd]) + "->" + str(new_prod_name[0]))
		compareYs(ysPref[product], ingreds, ing_map, db_file, prod_map)
		print("\n\n")


def printIngr(db_handler, id_ing):
	'''
	Function used for printing one specific ingredient

	:param db_handler: the database handler to be used for the interaction 
	:param id_ing: the ingredient ID to be searched.
	'''
	selIngQuery = "SELECT ingredient_name FROM ingredients WHERE ingredient_id=?"

	if db_handler != None:
		db_handler.execute(selIngQuery, (str(id_ing), ))
		ing_name = db_handler.fetchone()
		print("-Ing: " + str(ing_name))
	else:
		print("-Ing_ID: " + str(id_ing))

def compareYs(Y, yStar, ing_map, db_file, prod_map):
	'''
	Function used to compare two products. It shows which ingredients have been added and which have been removed

	:param Y: the base set of ingredients
	:param yStar: the modified set of ingredients
	:param ing_map: the ingredient mapping
	:param db_file: the database file to be used for the interaction
	:param prod_map: the products mappings.
	'''
	con = None
	db_handler = None

	diff = yStar - Y
	
	#Diff dict. [0] = added, [1] = removed, [2] = not touched
	differences = ([], [], [])

	if db_file:
		con = sqlite3.connect(db_file)
		db_handler = con.cursor()

	for x in range(len(diff)):
		if diff[x] == 1:
			differences[0].append(ing_map[x])
		elif diff[x]  == -1:
			differences[1].append(ing_map[x])
		else:
			differences[2].append(ing_map[x])
	print("[-] Differences Section")
	
	if len(differences[0]) == 0:
		print("[+] No ingredients added")
	else:
		print("[+] Addded ingredients section")
		for added in differences[0]:
			printIngr(db_handler, added)

	if len(differences[1]) == 0:
		print("[-] No ingredients removed")
	else:
		print("[-] Removed ingredients section")
		for removed in differences[1]:
			printIngr(db_handler, removed)

class Model:
    def __init__(self, nFeatures, nLabels, sess, model, layerSizes,nGdIter, lambda_dist=1):
        self.nFeatures = nFeatures
        self.nLabels = nLabels
        self.sess = sess
        self.model = model
        self.layerSizes = layerSizes

        self.trueY_ = tf.placeholder(tf.float32, shape=[None, nLabels], name='trueY')

        self.x_ = tf.placeholder(tf.float32, shape=[None, nFeatures], name='x')
        self.y0_ = tf.placeholder(tf.float32, shape=[None, nLabels], name='y')

        #Lambda value needed for the balance in the new preference inference problem
        self.lambda_dist = lambda_dist

        #Model selection, partial input convex or full input convex
        if model == 'picnn':
            f = self.f_picnn
        elif model == 'ficnn':
            f = self.f_ficnn

        #Set the proper energy function
        E0_ = f(self.x_, self.y0_, self.layerSizes)

        #Set learning rate and momentum
        lr = 0.01
        momentum = 0.9

        #Inference phase definition, since the energy function is concave here they use a gradient descent method.
        yi_ = self.y0_
        Ei_ = E0_
        vi_ = 0

        for i in range(nGdIter):
            prev_vi_ = vi_
            vi_ = momentum*prev_vi_ - lr*tf.gradients(Ei_, yi_)[0]
            yi_ = yi_ - momentum*prev_vi_ + (1.+momentum)*vi_
            Ei_ = f(self.x_, yi_, self.layerSizes, True)

        self.yn_ = yi_
        self.energies_ = Ei_


        #Formulation of the preference phase. It is basically the same mathematical model used in the previous phase but here we add 
        #a distance between the true Y and the predicted one, in this way the new product should not be too far from the original one.
        yi_ = self.y0_
        vi2_ = 0
        E1_ = self.energies_ + tf.reduce_mean(tf.square(self.trueY_ - yi_))
        #E1_ = E0_ 
        Ei_Loss_ = E1_
        for i in range(nGdIter):
            prev_vi2_ = vi2_
            vi2_ = momentum*prev_vi2_ - lr*tf.gradients(Ei_Loss_, yi_)[0]
            yi_ = yi_ - momentum*prev_vi2_ + (1.+momentum)*vi2_
            Ei_Loss_ = self.energies_ + self.lambda_dist * tf.reduce_mean(tf.square(self.trueY_ - yi_))
            #Ei_Loss_ = f(self.x_, yi2_, self.layerSizes, True) + self.lambda_dist * tf.reduce_mean(tf.square(self.trueY_ - yi2_))
        self.yn2_ = yi_

        #Calculate MSE
        self.mse_ = tf.reduce_mean(tf.square(self.yn_ - self.trueY_))

        self.opt = tf.train.AdamOptimizer(0.001)
        self.theta_ = tf.trainable_variables()
        self.gv_ = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta_)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv_)

        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]

        self.makeCvx = [v.assign(tf.abs(v)/10.) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        # for g,v in self.gv_:
        #     variable_summaries(g, 'gradients/'+v.name)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=0)    

    @staticmethod
    def iterate_mini_batches(inputs, targets, batch_size):
    	'''
    	@Author: Pedranza
    	Utility method to iterate over the training data in batches.
        At each call of the function, data are randomly shuffled and split in batches.
        An iterator over the batches is returned.
        :param inputs: X.
        :param targets: y.
        :param batch_size: Number of training examples for each batch.
        :return: Iterator over the batches, returning a couple (X, y) at each step.
        '''
    	assert len(inputs) == len(targets)
    	indices = np.arange(len(inputs))
    	np.random.shuffle(indices)
    	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    		excerpt = indices[start_idx:start_idx + batch_size]
    		yield inputs[excerpt], targets[excerpt]

    def save(self, path):
        '''	
    	Save the model in the specified path
    	:param self: self class
    	:param path: path where to save the mode
    	'''
        self.saver.save(self.sess, path)

    def load(self, path):
    	'''
    	Load the model from the specified path
		:param self: self class
		:param path: path from where to load the model
    	'''
    	self.saver.restore(self.sess, path)
    
    def trainNew(self, args, dataX, dataY, xs_valid, ys_valid):

	    save = args.save
	    nTrain = dataX.shape[0]

	    nIter = args.nEpoch

	    if(args.plot_acc=="yes"):
	        trainFields = ['iter', 'loss', 'accuracy']
	    else:
	        trainFields = ['iter', 'loss']
	    trainF = open(os.path.join(save, 'train.csv'), 'w')
	    trainW = csv.writer(trainF)
	    trainW.writerow(trainFields)

	    if(args.plot_acc == "yes"):
	        testFields = ['iter' , 'loss', 'accuracy']
	    else:
	        testFields = ['iter', 'loss']
	    testF = open(os.path.join(save, 'test.csv'), 'w')
	    testW = csv.writer(testF) 
	    testW.writerow(testFields)

	    self.trainWriter = tf.summary.FileWriter(os.path.join(save, 'train'),
	                                              self.sess.graph)
	    self.testWriter = tf.summary.FileWriter(os.path.join(save, 'test'),
	                                              self.sess.graph)
	    self.sess.run(tf.initialize_all_variables())
	    if not args.noncvx:
	        self.sess.run(self.makeCvx)

	    nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())

	    meta = {'nTrain': nTrain, 'nParams': nParams, 'nEpoch': args.nEpoch,
	    'Layer sizes:': args.layers, 'TestEpoch': args.testEpoch, 'model': args.model, 
	    'trainBatchSz': args.trainBatchSz, 'batch_trainON': args.batch_train,
	    'noncvx': args.noncvx}

	    if not args.data:
	        meta = {'nTrain': nTrain, 'nParams': nParams, 'nEpoch': args.nEpoch,
	        'Layer sizes:': args.layers, 'TestEpoch': args.testEpoch, 'model': args.model, 
	        'trainBatchSz': args.trainBatchSz, 'batch_trainON': args.batch_train, 'multilabel': args.multilabel, 'binarize': args.binarize, 'nsamples': args.n_samples,
	        'n_features': args.n_features, 'n_classes': args.n_classes,
	        'noncvx': args.noncvx}


	    metaP = os.path.join(save, 'meta.json')
	    with open(metaP, 'w') as f:
	        json.dump(meta, f, indent=2)

	    bestMSE = None
	    nErrors = 0
	    glob_index = 0
	    for i in range(nIter):
	        batch_n = 0
	        trainAcc = 0
	        mse_train = 0
	        tflearn.is_training(True)
	        start = time.time()
	        
	        if args.batch_train == "no":
	            print("=== Epoch {} ===".format(i)) 
	            y0 = np.full(dataY.shape, 0.5)
	            _, trainMSE, yN = self.sess.run(
	                [self.train_step, self.mse_, self.yn_],
	                feed_dict={self.x_: dataX, self.y0_: y0, self.trueY_: dataY})
	            print(" + loss: {:0.5e}".format(trainMSE))
	        else:
	            for batch in self.iterate_mini_batches(dataX, dataY, args.trainBatchSz):
	                print("=== Epoch {} (Batch {}/{}) ===".format(i, batch_n, np.ceil(nTrain/args.trainBatchSz)))
	                xBatch, yBatch = batch
	                y0 = np.full(yBatch.shape, 0.5)
	                _, trainMSE, yN = self.sess.run(
	                [self.train_step, self.mse_, self.yn_],
	                feed_dict={self.x_: xBatch, self.y0_: y0, self.trueY_: yBatch})

	                batch_acc, accuracies = util.macroAccuracy(yBatch, yN)
	                if args.plot_acc == "yes":
	                    trainAcc += batch_acc
	                    print("+train_accuracy: {:0.2e}".format(batch_acc))
	                mse_train += trainMSE
	                print("+mse_train: {:0.5e}".format(trainMSE))
	                if args.verbose == "yes":
	                    if args.plot_acc == "yes":
	                        trainW.writerow(((glob_index),trainMSE, batch_acc))
	                    else:
	                        trainW.writerow(((glob_index), trainMSE))
	                    trainF.flush()
	                batch_n += 1

	        glob_index += 1
	        
	        if not args.noncvx and len(self.proj) > 0:
	            self.sess.run(self.proj)

	        if args.plot_acc == "yes":
	            if args.batch_train == "yes":
	                trainAcc = trainAcc/np.ceil(nTrain/args.trainBatchSz)
	                trainMSE = mse_train/np.ceil(nTrain/args.trainBatchSz)
	                if args.verbose == "yes":
	                    trainW.writerow((glob_index, trainMSE, trainAcc))
	                else:
	                    trainW.writerow((i, trainMSE, trainAcc))
	            else:
	                trainAcc, accuracies = util.macroAccuracy(dataY, yN)
	                trainW.writerow((i, trainMSE, trainAcc))
	        else:
	            if args.batch_train == "yes":
	                if args.verbose == "yes":
	                    trainW.writerow((glob_index, trainMSE))
	                else:
	                    trainW.writerow((i, trainMSE))
	            else:
	                trainW.writerow((i, trainMSE))
	        trainF.flush()
	        print(" + batch training time: {:0.2f} s".format(time.time()-start))
	        glob_index += 1
	        if ((i+1) % args.testEpoch) == 0:
	            print("=== TESTING ===")
	            #Exit training
	            tflearn.is_training(False)
	            #Init predictions with max_entropy 
	            y0 = np.full(ys_valid.shape, 0.5)
	            #Predict
	            yN = self.sess.run(self.yn_, feed_dict={self.x_: xs_valid, self.y0_: y0})
	            testMse = util.macroSquaredError(ys_valid, yN)
	            print(" + testMSE: {:0.4e}".format(testMse))
	            if args.plot_acc == "yes":
	                testAcc, accuracies = util.macroAccuracy(ys_valid, yN)
	                print(" + testAcc: {:0.4f}".format(testAcc) )
	                testW.writerow((i, testMse, testAcc))
	            else:
	                testW.writerow((i, testMse))
	            testF.flush()
	            if args.save_accuracies == "yes":
	            	#util.saveAccuracies(ys_valid, yN, args.file_accuracies + "_" + str(i+1) + ":Epoch.results" )
	                util.saveMetrics(ys_valid, yN, args.file_accuracies + "_" + str(i+1) + ":Epoch.csv")
	    trainF.close()
	    testF.close()
	    os.system('./icnn.plot.py --workDir ' + save + ' --classification ' + args.plot_acc)



######################### PREFERENCE PART ###############################
    def preferenceBasedSelectionInteraction(self, X,Y,ing_map,ys_orig,scaler,nutr_map,args,prod_map, prodIDs):
    	'''
		Function used as first preference tests, using a qualitative analysis, more than a quantitative one.

		:param self: ICNN model.
		:param X: dataset input X
		:param Y: dataset output Y
		:param ing_map: ingredient mapppings.
		:param ys_orig: original, rescaled data
		:param scaler: scaler to be used for the normalization
		:param nutr_map: mapping for the nutrients.
		:param args: arguments passed to the script
		:param prod_map: mapping for the product IDs.
		:param prodIDs: product IDs to be used for the changements
		:return yStar: the predicted set of nutrients.
		:return modifiedNutr: the modified products.
    	'''

    	modifiedNutr = self.modifyNutr2(prodIDs, X, scaler, nutrToMod, nutr_map, args)

    	y0 = np.full(Y.shape, 0.5)
    	yStar = self.sess.run(self.yn2_, feed_dict={self.x_:modifiedNutr, self.y0_: y0, self.trueY_:Y})
    	yStar = (yStar >= 0.5).astype(np.int)

    	return (yStar, modifiedNutr) 

    def preferenceBasedSelection(self, X,Y,focus, ratios, focus_nutr, scaler):
    	'''
    	This function implements the inference for the new preference problem, the algorithm implemented is the following one:
    	1. Modify the selected products (X) using the specific function. Such a function modify the nutrients according to the focussed nutrients and the ratios to apply on them
    	2. Infere the new products using the new inferece problem, keeping into account both the minimization of the learned energy function and also the distance between the
    	new product (yStar) and the original one (Y)
    	3- Return the new product and the modified nutrients, which will be used from the comparison algorithms to measure the performances

    	:param X: the product set to use
    	:param Y: the ingredient set to use
    	:param focus: kind of focus used during the test over the nutrients.
    	:param ratios: ratios used for modifying the nutrient values.
    	:param focus_nutr: nutrients to put the focus on.
    	:param scaler: scaler used for the changements
    	:return yStar: the predicted set of ingredients
    	:return modifiedNutr: the modified products.
    	'''
    	modifiedNutr = self.modifyNutr3(X, focus, ratios, focus_nutr, scaler)
    	if modifiedNutr.ndim == 1:
    		modifiedNutr = np.asmatrix(modifiedNutr)
    	if Y.ndim == 1:
    		Y = np.asmatrix(Y)
    	

    	y0 = np.full(Y.shape, 0.5)
    	yStar = self.sess.run(self.yn2_, feed_dict={self.x_:modifiedNutr, self.y0_: y0, self.trueY_:Y})

    	yStar = (yStar >= 0.5).astype(np.int)

    	return (yStar, modifiedNutr)  

    def modifyNutr3(self, X, focus, ratios, focus_nutr, scaler):
    	'''
    	This functions modify the nutrients values (only on the focused ones), according to the values specified as ratios.

    	:param self: ICNN model
    	:param X: the input target dataset
    	:param focus: the kind of focus [all | part_of]
    	:param ratios: the ratios to be used for the changes on the nutrients
    	:param focus_nutr: the nutrients to put the focus on.
    	:param scaler: the scaler used for the dataset normalization
    	:return: the modified set of products.
    	'''
    	orig_data = scaler.inverse_transform(X)
    	tmp = orig_data
    	if focus == "all":
    		focus_nutr = range(38)

    	k = 0
    	if X.ndim == 1:
    		for id in focus_nutr:
    			if len(ratios) == 1:
    				mod_nutr = orig_data[id] * ratios[0]
    			else:
    				mod_nutr = orig_data[id] * ratios[k]
    				k = k + 1
    			orig_data[id] = mod_nutr
    	else:
    		for i in range(len(X)):
    			k = 0
    			for id in focus_nutr:
    				if len(ratios) == 1:
	    				mod_nutr = orig_data[i][id]*ratios[k]
	    				k = 0
	    			else:
	    				mod_nutr = orig_data[i][id]*ratios[k]
	    				k = k + 1
	    			orig_data[i][id] = mod_nutr



    	toRet = scaler.transform(orig_data)
    	return scaler.transform(orig_data)

    	
    def modifyNutr2(self, prodIDs, samples, scaler, nutrIDs, nutrMap, args):
    	'''
		Function used for initial quantitative tests togheter with preferenceBasedSelectionInteraction. This function modify a set of products according to the
		values specified by the user.

		:param self: ICNN model
		:param prodIDs: the preference set of product IDs to modify
		:param samples: the preference set of products to modify
		:param scaler: the scaler used for the normalization
		:param nutrIDs: the nutrient IDs to put the focus on, for the changes
		:param nutrMap: the nutrient mapping
		:param args: the arguments passed to the main script

		:return the modified products
    	'''
    	scaled_data = scaler.inverse_transform(samples)
    	con = sqlite3.connect(args.db_file)
    	db_handler = con.cursor()
    	selProdQuery = "SELECT name FROM products WHERE ndbno=?"
    	selNutrQuery = "SELECT nutrient_name FROM  nutrients WHERE nutrient_id=?"

    	for i in range(len(samples)):
    		db_handler.execute(selProdQuery, (int(prodIDs[i]),))
    		prod_name = db_handler.fetchone()
    		print("#### PRODUCT #" + str(prodIDs[i]) + "->" + str(prod_name[0]) +  " ####")
    		for id in nutrIDs:
    			db_handler.execute(selNutrQuery, (int(nutrMap[id]), ))
    			nutr_name = db_handler.fetchone()
    			print("# Nutrient " + str(nutr_name[0]) + " Value: " + str(scaled_data[i][id]))
    			modVal = input("Enter the new value for the nutrient: ")
    			scaled_data[i][id] = modVal

    	return scaler.transform(scaled_data)

    def compareIngred(self, y_base, y_pred):
    	'''
		Function used to compare two different set  of ingredients, the base one and the predicted one.

		:param self: the ICNN model
		:param y_base: the base set of ingredients
		:param y_pred: the predicted set of ingredients
    	'''
    	diff = np.subtract(y_pred, y_base)
    	print("tot SIZE")
    	print(y_base.size)
    	print("tot prod")
    	print(y_base.shape[0])
    	
    	return (np.count_nonzero(diff) / y_base.size)


#######################################################################
    def f_ficnn(self, x, y, layerSizes, reuse=False):
        fc = tflearn.fully_connected
        xy = tf.concat((x, y), axis=1)

        if layerSizes[-1] != self.nLabels:
            print("Appending the labels to layer sizes")
            layerSizes.append(self.nLabels)

        prevZ = None
        for i, sz in enumerate(layerSizes):
            z_add = []

            with tf.variable_scope('z_x{}'.format(i)) as s:
                z_x = fc(xy, sz, reuse=reuse, scope=s, bias=True)
                z_add.append(z_x)

            if prevZ is not None:
                with tf.variable_scope('z_z{}_proj'.format(i)) as s:
                    z_z = fc(prevZ, sz, reuse=reuse, scope=s, bias=False)
                    z_add.append(z_z)

            if sz != 1:
                z = tf.nn.relu(tf.add_n(z_add))
            prevZ = z

        return tf.contrib.layers.flatten(z)

    def f_picnn(self, x, y, layerSizes, reuse=False):
        fc = tflearn.fully_connected
        xy = tf.concat((x, y), axis=1)

        if self.layerSizes[-1] != self.nLabels:
            print("Appending the labels to layer sizes")
            self.layerSizes.append(self.nLabels)

        prevZ, prevU = None, x
        for layerI, sz in enumerate(layerSizes):
            if sz != 1:
                with tf.variable_scope('u'+str(layerI)) as s:
                    u = fc(prevU, sz, scope=s, reuse=reuse)
                    u = tf.nn.relu(u)

            z_add = []

            if prevZ is not None:
                with tf.variable_scope('z{}_zu_u'.format(layerI)) as s:
                    prevU_sz = prevU.get_shape()[1].value
                    zu_u = fc(prevU, prevU_sz, reuse=reuse, scope=s,
                            activation='relu', bias=True)
                with tf.variable_scope('z{}_zu_proj'.format(layerI)) as s:
                    z_zu = fc(tf.multiply(prevZ, zu_u), sz, reuse=reuse, scope=s,
                                bias=False)
                z_add.append(z_zu)

            with tf.variable_scope('z{}_yu_u'.format(layerI)) as s:
                yu_u = fc(prevU, self.nLabels, reuse=reuse, scope=s, bias=True)
            with tf.variable_scope('z{}_yu'.format(layerI)) as s:
                z_yu = fc(tf.multiply(y, yu_u), sz, reuse=reuse, scope=s, bias=False)
            z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(layerI)) as s:
                z_u = fc(prevU, sz, reuse=reuse, scope=s, bias=True)
            z_add.append(z_u)

            z = tf.add_n(z_add)
            if sz != 1:
                z = tf.nn.relu(z)

            prevU = u
            prevZ = z

        return tf.contrib.layers.flatten(z)

if __name__=='__main__':
    main()


