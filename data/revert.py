'''
Script reverting a dataset, output will be input and input will be output, used for the ICNN. 
'''
import sys
import pickle

objects = []
with (open(sys.argv[1], "rb")) as openfile:
	while True:
		try:
			objects = pickle.load(openfile)
		except EOFError:
			break


reverted = { 'xs' : objects['ys'],
			 'ys' : objects['xs'],
			 'products_map' : objects['products_mapping'],
			 'ingred_map' : objects['ingredients_mapping'],
			 'ys_orig': objects['ys_original'],
			 'scaler': objects['scaler'],
			 'nutrients_mapping': objects['nutrients_mapping'] 
			}

with open(sys.argv[1] + ".reverted.pickle", "wb") as output_file:
	pickle.dump(reverted, output_file)