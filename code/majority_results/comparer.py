'''
Script used to compare the results given from two models, in this case the baseline and the ICNN one. 
It needs two CVS file containing accuracy, precision, recall and f1 for the model. Then it compare the metrics on the same classes,
 creating an additional CSV file containing the metrics for both model and the differences between the second one and the first. Moreover,
 since it has been used for Tasteme, I've implemented a threshold to be used over the F1 difference, if the difference is greater then the threshold,
  the script interacts with the DB file to retrieve the ingredients name related to such comparison. 
 In this way I've identified the most-learned ingredients.
'''
import csv
import itertools
import argparse
import sqlite3
import pickle


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--baseline', type=str, default='work/accuracies.results')
	parser.add_argument('--toCompare', type=str, default='work2/accuracies.results')
	parser.add_argument('--toSave', type=str, default='work2/compare.result')
	parser.add_argument('--f1_threshold', type=float, default=0.10)
	parser.add_argument('--db_file', type=str)
	parser.add_argument('--data', type=str)
	args = parser.parse_args()
    
	f1 = open(args.baseline)
	f2 = open(args.toCompare)

	csv_f1 = csv.reader(f1)
	csv_f2 = csv.reader(f2)

	con = None
	db_handler = None
	selIng = "SELECT ingredient_name FROM ingredients WHERE ingredient_id=?"

	with open(args.data, "rb") as f:
		data = pickle.load(f)
		ing_map = data["ingred_map"]

	i=0
	writeQueue = []
	with open(args.baseline) as f1, open(args.toCompare) as f2:
		r1=csv.reader(f1)
		r2=csv.reader(f2)
		for v1, v2 in zip(r1, r2):

			
			baseLine = float(v1[0])
			toCompare = float(v2[0])
			diff = toCompare - baseLine
			
			precBase = float(v1[1])
			recBase = float(v1[2])
			f1Base = float(v1[3])

			precModel = float(v2[1])
			recModel = float(v2[2])
			f1Model = float(v2[3])

			f1Diff = f1Model - f1Base
			if float(f1Diff) > float(args.f1_threshold):
				con = sqlite3.connect(args.db_file)
				db_handler = con.cursor()
				db_handler.execute(selIng, (ing_map[i], ))
				ing_name = db_handler.fetchone()
				
				toWrite = (i, baseLine, toCompare, diff,precBase, precModel, recBase, recModel, f1Base, f1Model, f1Diff, ing_name)
			else:
				toWrite = (i, baseLine, toCompare, diff,precBase, precModel, recBase, recModel, f1Base, f1Model, f1Diff)
			writeQueue.append(toWrite)
			i = i+1

	with open(args.toSave, 'w') as f:
		writer = csv.writer(f)
		header = ("ingredient", "baseline_acc", "icnn_acc", "diff_acc", "prec_base", "prec_icnn", "recBase", "recModel", "f1Base", "f1Model", "f1Diff", "ingred_name")
		writer.writerow(header)
		for row in writeQueue:
			writer.writerow(row)


if __name__=='__main__':
    main()
