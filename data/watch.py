import sys
import pickle

objects = []
with (open(sys.argv[1], "rb")) as openfile:
	while True:
		try:
			objects = pickle.load(openfile)
		except EOFError:
			break

for obj in objects:
	print(obj)
	print(objects[obj])
	print("\n\n")

'''
for obj in objects:
	print("Lenght obj:" + str(len(obj[sys.argv[2]])))
	print("Lenght features:" + str(len(obj[sys.argv[2]][0])))
	#print(obj[sys.argv[2]])
	print("\n")
'''