'''
Script used to creaate different classification/regression synthetic datasets to be used by the ICNN in order to test it.
'''
import sys
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--classific', type=str, default='yes')
parser.add_argument('--samp', type=int, default=8000)
parser.add_argument('--n_feat', type=int, default=30)

parser.add_argument('--n_inf', type=int, default=20)
parser.add_argument('--n_red', type=int, default=5)
parser.add_argument('--n_class', type=int, default=5)
parser.add_argument('--save', type=str, default="data_synth.pickle")
args = parser.parse_args()

if args.classific == "yes":
	n_clust= args.n_inf / args.n_class
	#make dataset
	xs, ys = make_classification(n_samples=args.samp, n_features=args.n_feat, n_informative=args.n_inf, n_redundant=args.n_red, n_classes=args.n_class, n_clusters_per_class=round(n_clust))

	#Making 1-of-hot encodings
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(ys)
	integer_encoded = ys.reshape(len(ys), 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
else:

	xs,ys = make_regression(n_samples=args.samp, n_features=args.n_feat, n_informative=args.n_inf, n_targets=args.n_class)
	onehot_encoded = ys

dataset = { 'xs' : xs,
			 'ys' : onehot_encoded,
			}

with open(args.save, "wb") as output_file:
	pickle.dump(dataset, output_file)
print("synth dataset created under " + args.save)
