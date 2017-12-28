#!/usr/bin/env python3
'''
This script contains plotting functions used from the main ICNN script file.
'''
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np
import pandas as pd
import math

import os
import sys
import json
import glob

scriptDir = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workDir', type=str, default="work")
    parser.add_argument("--classification", type=str, default="no")
    # parser.add_argument('--ymin', type=float, default=1e-4)
    # parser.add_argument('--ymax', type=float, default=1e-1)
    args = parser.parse_args()

    trainF = os.path.join(args.workDir, 'train.csv')
    if os.path.isfile(trainF):
        trainDf, testDf, meta = getDataSingle(args.workDir)
    else:
        assert(False)

    if args.classification=="yes":
        plotMSE(trainDf, testDf, meta, args.workDir)
        plotMeanAcc(trainDf, testDf, meta, args.workDir)
    else:
        plotMSE(trainDf, testDf, meta, args.workDir)

def getDataSingle(workDir):
    '''
    Retrieve all the data needed for the plotting functions

    :param workDir: the working directory
    '''
    trainF = os.path.join(workDir, 'train.csv')
    testF = os.path.join(workDir, 'test.csv')
    metaF = os.path.join(workDir, 'meta.json')

    trainDf = pd.read_csv(trainF, sep=',')
    testDf = pd.read_csv(testF, sep=',')

    with open(metaF, 'r') as f:
        meta = json.load(f)

    return trainDf, testDf, meta

def plotAcc(trainDf, testDf, meta, workDir):
    '''
    Plot the accuracy of both train and test sets

    :param trainDf: cvs training file
    :param testDf: csv test file
    :param meta: metadata
    :workDir: working directory
    '''
    nTrain = meta['nTrain']
    trainBatchSz = meta['trainBatchSz']

    # fig, ax = plt.subplots(1, 1, figsize=(5,2))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.25,left=0.15) # For (5, 2)
    fig.subplots_adjust(bottom=0.1,left=0.1)
    N = math.ceil(nTrain/trainBatchSz)

    trainIters = trainDf['iter'].values
    trainF1s = trainDf['mean_accuracy'].values

    trainIters = trainIters[N:]/np.ceil(nTrain/trainBatchSz)
    trainF1s = [sum(trainF1s[i-N:i])/N for i in range(N, len(trainF1s))]
    plt.plot(trainIters, trainF1s, label='Train')

    testIters = testDf['iter'].values
    testF1s = testDf['mean_accuracy'].values
    if len(testF1s) > 0:
        plt.plot(testIters/np.ceil(nTrain/trainBatchSz), testF1s, label='Test')
    # trainP = testDf['f1'].plot(ax=ax)
    plt.xlabel("Epoch")
    plt.ylabel("mean_accuracy")
    plt.ylim(ymin=0, ymax=1)
    plt.xlim(xmin=0)
    # plt.xlim(xmin=0,xmax=50)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    # ax.set_yscale('log')
    plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "mean_acc."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

def plotLoss(trainDf, testDf, meta, workDir):
    '''
    Plot the loss of both train and test sets

    :param trainDf: cvs training file
    :param testDf: csv test file
    :param meta: metadata
    :workDir: working directory
    '''
    nTrain = meta['nTrain']
    trainBatchSz = meta['trainBatchSz']

    # fig, ax = plt.subplots(1, 1, figsize=(5,2))
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.25,left=0.15) # For (5, 2)
    fig.subplots_adjust(bottom=0.1,left=0.1)
    N = math.ceil((10.0/10.0)*nTrain/meta['trainBatchSz'])

    trainIters = trainDf['iter'].values
    trainLoss = trainDf['loss'].values

    trainIters = trainIters[N:]/np.ceil(nTrain/trainBatchSz)
    trainLoss = [sum(trainLoss[i-N:i])/N for i in range(N, len(trainLoss))]
    plt.plot(trainIters, trainLoss, label='Train')

    if 'loss' in testDf:
        testIters = testDf['iter'].values
        testLoss = testDf['loss'].values
        if len(testLoss) > 0:
            plt.plot(testIters/np.ceil(nTrain/trainBatchSz), testLoss, label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    # plt.xlim(xmin=0,xmax=50)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
    plt.legend()
    ax.set_yscale('log')
    # plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "loss."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

def plotMSE(trainDf, testDf, meta, workDir):
    '''
    Plot the MSE of both train and test sets

    :param trainDf: cvs training file
    :param testDf: csv test file
    :param meta: metadata
    :workDir: working directory
    '''
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.25,left=0.15) # For (5, 2)
    fig.subplots_adjust(bottom=0.1,left=0.1)
    trIter = trainDf['iter'].values
    trLoss = trainDf['loss'].values
    plt.plot(trIter, trLoss, label='Train')

    testIter = testDf['iter'].values
    testLoss = testDf['loss'].values
    plt.plot(testIter, testLoss, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "loss."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

def plotMeanAcc(trainDf, testDf, meta, workDir):
    '''
    Plot the Mean accuracy (since multilabel predictions) of both train and test sets

    :param trainDf: cvs training file
    :param testDf: csv test file
    :param meta: metadata
    :workDir: working directory
    '''
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.25,left=0.15) # For (5, 2)
    fig.subplots_adjust(bottom=0.1,left=0.1)
    trIter = trainDf['iter'].values
    trLoss = trainDf['accuracy'].values
    plt.plot(trIter, trLoss, label='Train')

    testIter = testDf['iter'].values
    testLoss = testDf['accuracy'].values
    plt.plot(testIter, testLoss, label="Test")
    plt.xlabel("Iter")
    plt.ylabel("Mean_Accuracy")
    plt.legend()
    for ext in ['pdf', 'png']:
        f = os.path.join(workDir, "mean_acc."+ext)
        fig.savefig(f)
        print("Created {}".format(f))

if __name__ == '__main__':
    main()
