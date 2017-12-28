'''
Converts a pickle into a csv file. 
'''
import os 
import pickle 
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pickle_file', type=str)
parser.add_argument("--csv_file" , type=str, default="eth1")
args = parser.parse_args()


with open(args.pickle_file, "rb") as file:
	data = pickle.load(file)

xs = data["xs"][100000]
ys = data["ys"][100000]

np.savetxt(args.csv_file, xs, delimiter=',')
