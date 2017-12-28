'''
Take a subset of the specified pickle dataset file.
'''
import sys
import pickle

objects = []
num_objects = sys.argv[2]
new_file = sys.argv[3]

with (open(sys.argv[1], "rb")) as openfile:
	while True:
		try:
			objects.append(pickle.load(openfile))
		except EOFError:
			break


new_obj = { 'xs' : objects[0]['xs'][:int(num_objects)],
			 'ys' : objects[0]['ys'][:int(num_objects)],
			}

with open(new_file, "wb") as output_file:
	pickle.dump(new_obj, output_file)