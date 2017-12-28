'''
Script used to merge all the datasets created with the ICNNs output and the enhanced nutrients
'''

import os
import pickle
import numpy as np
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', nargs="+", type=str, help="List of pickle files to be merged")
	parser.add_argument('--out', type=str, default="data.pickle")
	args = parser.parse_args()
	
	first = True
	i = 0
	print("[-] Starting loading data...")
	for dataset in args.data:
		print("[--] Appending dataset {} - {}/{}".format(dataset, str(i), str(len(args.data))))
		with open(dataset, "rb") as file:
			data = pickle.load(file)
			if first:
				to_ret_xs = data["xs"]
				to_ret_ys = data["ys"]
				first = False
			else:
				to_ret_xs = np.append(to_ret_xs, data["xs"], axis=0)
				to_ret_ys = np.append(to_ret_ys, data["ys"], axis=0)
		i = i + 1

	print("[-] Dumping output file...")
	with open(args.out, "wb") as out_file:
		new_data = {"xs":to_ret_xs, "ys":to_ret_ys}
		pickle.dump(new_data, out_file)
	print("[-] Output file dumped!")

if __name__=='__main__':
    main()