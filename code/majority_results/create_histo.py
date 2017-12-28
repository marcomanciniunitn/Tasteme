'''
script used to crate the histograms related to the frequencies of the ingredients in the dataset.
'''
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--from_file', type=str, default='freq.unnormalized.count')
parser.add_argument('--to', type=str, default="freq.unnormalized.hist")
args = parser.parse_args()

elems = []
x = []
y = []
with open(args.from_file, 'rb') as handler:
	for line in handler:
		el = line.split()
		x.append(int(el[0]))
		y.append(float(el[1]))
	print("--Frequencies loaded!")

plt.bar(x,y) 
plt.xlabel('Ingredients')
plt.ylabel("Frequency")
'''
for i in range(len(y)):
	plt.hlines(y[i], 0, x[i])
'''
plt.show()
