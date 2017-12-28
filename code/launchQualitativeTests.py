#!/usr/bin/env python3

'''
import matplotlib as mpl
from matplotlib import cm
# mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('bmh')

fig, ax = plt.subplots(1, 1, figsize=(1, 8))
fig.subplots_adjust(bottom=0.05,top=0.95,left=0,right=0.65)
norm = mpl.colors.Normalize(vmin=0., vmax=1.)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cm.bwr,
                                norm=norm,
                                orientation='vertical', alpha=0.5)

print("Created legend.{pdf,png}")
plt.savefig('legend.pdf')
plt.savefig('legend.png')
'''
import os
import sys 
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--lambda_val', type=str, default='1', help="Lambda value for energy-inference trade-off")
parser.add_argument('--nutrients', type=int, nargs="+", default=[17,24,33,35,37], help="Nutrients to play on")
parser.add_argument('--new_values', type=float, nargs="+", default=[1], help="Rateo values to use")
parser.add_argument("--save_dir", type=str, default="/home/doomdiskday/MEGAsync/result-PRO/toKeepSaved")
args = parser.parse_args()

test_dir_name = "part_"
for nutr in args.nutrients:
	test_dir_name = test_dir_name + str(nutr) + "-"

test_dir_name = test_dir_name + "-"

for rateo in args.new_values:
	if rateo != args.new_values[-1]:
		test_dir_name = test_dir_name + str(rateo) + "x-"
	else:
		test_dir_name = test_dir_name + str(rateo) + "x"

final_cmd = "python3.5 icnn_multilabel.py --save /home/doomdiskday/MEGAsync/result-PRO/toKeepSaved --data ../data/nutr_ingredient/data_also_original.pickle.reverted.pickle --save_model no --path_model 'toKeepSaved/save/' --db_file ../data/nutr_ingredient/data_no_filters.db --test_dir "

final_cmd = final_cmd + test_dir_name + " --test_ratio "

for rate in args.new_values:
	final_cmd = final_cmd + str(rate) + " "

final_cmd = final_cmd + "--test_type incremental --test_nutrients "
tmp = final_cmd
for i in range(38):
	final_cmd = final_cmd + str(i) + " "
	final_cmd = final_cmd + "--test_focus part_of --test_lambda " + args.lambda_val + " > /tmp/tmp.res"
	print("Final cmd: " + str(final_cmd))
	os.system(final_cmd)
	print("Info for nutrient " + str(i) + ": ")
	os.system("tail -6 /tmp/tmp.res > /home/doomdiskday/MEGAsync/result-PRO/toKeepSaved/" + str(i) + ".res")
	final_cmd = tmp

'''	
for nutr in args.nutrients:
	final_cmd = final_cmd + str(nutr) + " "

final_cmd = final_cmd + "--test_focus part_of --test_lambda " + args.lambda_val

os.system(final_cmd)
'''

