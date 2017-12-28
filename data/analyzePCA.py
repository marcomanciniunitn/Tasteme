'''
Script used for some data analysis performed over the dataset. PCA has been applied to look for informative features, the cardinality has also been calculated to understand
the average number of ingredients per product. Finally I've also implemented a bucketization over the ingredients frequencies, so to see the dispersion of the ingredients.
'''
import os
import argparse
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--components', type=int, default=4)
parser.add_argument('--data', type=str, default="")
parser.add_argument('--in_out', type=str, default="in")
parser.add_argument("--cardinalities", type=str, default="cardinalities.pickle")
parser.add_argument("--threshold_card", type=int, default=2)
parser.add_argument("--bucketsize", type=int, default=5)

args = parser.parse_args()

with open(args.data, 'rb') as f:
    data = pickle.load(f)
    (dataX, dataY) = data["xs"], data["ys"]

pca = PCA(n_components=args.components)

if args.in_out == "in":
	pca.fit(dataX)
else:
	pca.fit(dataY)

print("----------PCA RESULTSET-----------\n")
print("--Explained variance ratio:")
print(pca.explained_variance_ratio_)


c = np.count_nonzero(dataY)
cardin = c / (float(dataY.shape[0]) * float(dataY.shape[1]))
print("----------CARDINALITY RESULTSET-----------")
print("--Cardinality: " + str(cardin))

with open(args.cardinalities, "rb") as f:
	card = pickle.load(f)

c = 0
for x in card:
	if card[x] >= args.threshold_card:
		c = c+1

perc = c * 100.0 / float(len(card))
print("--Percentage of dataset with more than " + str(args.threshold_card) + " ingredients:  " + str(perc))

nBuckets = np.ceil(float(dataY.shape[1]) / args.bucketsize)

buckets = {}
c = 0

for x in card:

	buck = int(card[x]/args.bucketsize)

	#print(str(card[x]) + " aaa " + str(buck))
	if buck not in buckets.keys():
		buckets[buck] = 1
	else:
		buckets[buck] += 1

#print(buckets)

x,y = zip(*buckets.items())
plt.bar(x,y) 
plt.xlabel('Buckets [' + str(args.bucketsize) + "]")
plt.ylabel("Frequency")
plt.show()

#decomment if needed to recreate cardinalities
'''
c=0
cardinalities = {}
i = 0
for x in range(dataY.shape[0]):
	for y in range(dataY.shape[1]):
		if dataY[x][y] == 1:
			c = c + 1
	cardinalities[i] = c
	i = i+1
	c = 0

with open(args.cardinalities, "wb") as f:
	pickle.dump(cardinalities, f)

print(cardinalities)

'''

