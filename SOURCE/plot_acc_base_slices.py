import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m

def listOfCsvFileForPlotAccBaseSlice(dataset):
	if dataset == 'orl' or dataset == 'ar':
		file_dir = os.listdir(f'./results/{dataset}')
		csv_file_dir = []

		for fd in file_dir:
			if 'csv' in fd:
				s_sl_e = np.array(re.findall('\d+', file_dir), dtype=int)
				if s_sl_e[0] == 5 and 'a_part_of' not in fd:
					csv_file_dir.append(fd)

		return csv_file_dir


csv_files_lst_ar = listOfCsvFileForPlotAccBaseSlice('ar') # csv files list - the result of ar face database after training SISA
csv_files_lst_orl = listOfCsvFileForPlotAccBaseSlice('orl') # csv files list - the result of orl dataset after training SISA
color_sl = ['black', '#013220', '#545312', '#C4A484', '#CBC3E3', '#ADD8E6'] # color arrays for plot
base_sl_color_i = 0 # index of color arrays
max_acc, min_acc = 0, np.iinfo(np.int32).max # for xticks, yticks, zticks

# ======== Data visulization - AR Face Database ========
csfont = {'fontname':'serif'}
fig, ax = plt.subplots(1,2, figsize=(19,5))
for csv_sl in csv_files_lst_ar:
	csv_name_info = np.array(re.findall('\d+', file_dir), dtype=int)
	df = pd.read_csv(csv_S)
	x = df['nb_epochs'].values
	y = df['accuracy'].values * 100
	ax[0].plot(x, y,  color=color_sl[base_sl_color_i], label=f"{csv_name_info[1]}")


	if x.max() > max_nb_epochs:
		max_nb_requests = x.max()
	if x.min() < min_nb_epochs:
		min_nb_requests = x.min()
	if y.max() > max_acc:
		max_acc = z.max()
	if y.min() < min_acc:
		min_acc = z.min()

	base_sl_color_i += 1


max_acc, min_acc =  int(round(max_acc,0)) + 1, int(m.floor(min_acc)) - 1
step_acc = max_acc - min_acc

if step_acc == 0:
	step_acc = 2
else:
	step_acc = step_acc // 5

ax[0].set_xticks(np.arange(0,26,5))
ax[0].set_yticks(np.arange(min_acc, max_acc, step_acc))
ax[0].set_xlabel('Số kỷ nguyên', fontsize=15, **csfont)
ax[0].set_ylabel('Độ chính xác (s)', fontsize=15, **csfont)
ax[0].set_title('AR Face Database (theo số lát cắt)', fontsize=20, fontweight="bold", **csfont)
ax[0].legend( fontsize=15, title='Số lát cắt', loc='lower right', prop={'family': 'serif'})


# reset to continue visualize another diagram - ORL Dataset
max_acc, min_acc = 0, np.iinfo(np.int32).max
base_sl_color_i = 0 # index of color arrays

# ======== Data visulization - ORL ========
for csv_sl in csv_files_lst_ar:
	csv_name_info = np.array(re.findall('\d+', file_dir), dtype=int)
	df = pd.read_csv(csv_S)
	x = df['nb_epochs'].values
	y = df['accuracy'].values * 100
	ax[1].plot(x, y,  color=color_sl[base_sl_color_i], label=f"{csv_name_info[1]}")


	if x.max() > max_nb_epochs:
		max_nb_requests = x.max()
	if x.min() < min_nb_epochs:
		min_nb_requests = x.min()
	if y.max() > max_acc:
		max_acc = z.max()
	if y.min() < min_acc:
		min_acc = z.min()

	base_sl_color_i += 1


max_acc, min_acc =  int(round(max_acc,0)) + 1, int(m.floor(min_acc)) - 1
step_acc = max_acc - min_acc

if step_acc == 0:
	step_acc = 2
else:
	step_acc = step_acc // 5

ax[1].set_xticks(np.arange(0,51,10))
ax[1].set_yticks(np.arange(min_acc, max_acc, step_acc))
ax[1].set_xlabel('Số kỷ nguyên', fontsize=15, **csfont)
ax[1].set_ylabel('Độ chính xác (s)', fontsize=15, **csfont)
ax[1].set_title('ORL (theo số lát cắt)', fontsize=20, fontweight="bold", **csfont)
ax[1].legend(fontsize=15, title='Số lát cắt', loc='lower right', prop={'family': 'serif'})

plt.savefig('face_plot_base_on_slices.png')
plt.show()