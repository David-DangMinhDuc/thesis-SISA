import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m

def listOfCsvFileForPlotAccBaseShard(dataset):
	if dataset == 'orl' or dataset == 'ar':
		file_dir = os.listdir(f'./results/{dataset}')
		csv_file_dir_base_S = []
		csv_file_dir_base_a_part_of_S = []
		for fd in file_dir:
			if 'csv' in fd:
				s_sl_e = np.array(re.findall('\d+', file_dir), dtype=int)
				if s_sl_e[1] == 1:
					if 'a_part_of' in fd:
						csv_file_dir_base_a_part_of_S.append(fd)
					else:
						csv_file_dir_base_S.append(fd)

		return csv_file_dir_base_S, csv_file_dir_base_a_part_of_S


csv_files_lst_ar_S, csv_files_lst_ar_part_of_S = listOfCsvFileForPlotAccBaseShard('ar') # csv files list - the result of ar face database after training SISA
csv_files_lst_orl_S, csv_files_lst_orl_part_of_S = listOfCsvFileForPlotAccBaseShard('orl') # csv files list - the result of orl dataset after training SISA
color_S, color_part_of_S = ['#ADD8E6', 'blue', 'purple'], ['orange', 'red', 'brown'] # color arrays for plot
base_S_color_i, base_a_part_of_S_color_i = 0, 0 # index of color arrays
max_nb_requests, min_nb_requests, max_time, min_time, max_acc, min_acc = 0, np.iinfo(np.int32).max,
																		0, np.iinfo(np.int32).max,
																		0,np.iinfo(np.int32).max # for xticks, yticks, zticks

# ======== Data visulization - AR Face Database ========
csfont = {'fontname':'serif'}
fig, ax = plt.subplots(1,2, figsize=(15,15), subplot_kw=dict(projection="3d"))
for csv_S in csv_files_lst_ar_S:
	csv_name_info = np.array(re.findall('\d+', file_dir), dtype=int)
	df = pd.read_csv(csv_S)
	x = df['nb_requests'].values
	y = df['retraining_time'].values
	z = df['accuracy'].values * 100
	if csv_name_info[0] != 1:
		ax[0].plot(x, y, z, marker='o', color=color_S[base_S_color_i], label=f"SISA (S = {csv_name_info[0]})")
	else:
		ax[0].plot(x, y, z, marker='D', color='black', label=f"Batch K=") # K là batch size m chọn với AR Face Database

	if x.max() > max_nb_requests:
		max_nb_requests = x.max()
	if x.min() < min_nb_requests:
		min_nb_requests = x.min()
	if y.max() > max_time:
		max_time = y.max()
	if y.min() < min_time:
		min_time = y.min()
	if z.max() > max_acc:
		max_acc = z.max()
	if z.min() < min_acc:
		min_acc = z.min()

	base_S_color_i += 1

for csv_a_part_of_S in csv_files_lst_ar_part_of_S:
	csv_name_info = np.array(re.findall('\d+', file_dir), dtype=int)
	df = pd.read_csv(csv_S)
	x = df['nb_requests'].values
	y = df['retraining_time'].values
	z = df['accuracy'].values * 100
	ax[0].plot(x, y, z, marker='o', color=color_part_of_S[base_a_part_of_S_color_i], label=f"SISA (S = 1/{csv_name_info[0]})")
	
	if x.max() > max_nb_requests:
		max_nb_requests = x.max()
	if x.min() < min_nb_requests:
		min_nb_requests = x.min()
	if y.max() > max_time:
		max_time = y.max()
	if y.min() < min_time:
		min_time = y.min()
	if z.max() > max_acc:
		max_acc = z.max()
	if z.min() < min_acc:
		min_acc = z.min()

	base_a_part_of_S_color_i += 1


max_nb_requests, min_nb_requests, max_time, min_time, max_acc, min_acc = int(round(max_nb_requests,0)) + 1, m.floor(min_nb_requests) - 1,  
																		int(round(max_time,0)) + 1, m.floor(min_time) - 1, 
																		int(round(max_acc,0)) + 1, int(m.floor(min_acc)) - 1
step_nb_requests, step_time, step_acc = max_nb_requests - min_nb_requests, max_time - min_time, max_acc - min_acc

if step_time == 0 and step_acc == 0:
	step_nb_requests, step_time, step_acc = 1, 1, 1
else:
	step_nb_requests, step_time, step_acc = step_nb_requests // 4, step_time // 4, step_acc // 4

ax[0].set_xticks(np.arange(min_nb_requests, max_nb_requests, step_nb_requests))
ax[0].set_yticks(np.arange(min_time, max_time, step_time))
ax[0].set_zticks(np.arange(min_acc, max_acc, step_acc))

ax[0].set_xlabel('Số yêu cầu', fontsize=15, **csfont)
ax[0].set_ylabel('Thời gian phân tích (s)', fontsize=15, **csfont)
ax[0].set_zlabel('Độ chính xác (%)', fontsize=15, **csfont)
ax[0].set_title('AR Face Database (theo số phân đoạn)', fontsize=20, fontweight="bold", **csfont)


# reset to continue visualize another diagram - ORL Dataset
max_nb_requests, min_nb_requests, max_time, min_time, max_acc, min_acc = 0, np.iinfo(np.int32).max,
																		0, np.iinfo(np.int32).max,
																		0,np.iinfo(np.int32).max 

base_S_color_i, base_a_part_of_S_color_i = 0, 0 # index of color arrays

# ======== Data visulization - ORL ========
for csv_S in csv_files_lst_orl_S:
	csv_name_info = np.array(re.findall('\d+', file_dir), dtype=int)
	df = pd.read_csv(csv_S)
	x = df['nb_requests'].values
	y = df['retraining_time'].values
	z = df['accuracy'].values * 100
	if csv_name_info[0] != 1:
		ax[1].plot(x, y, z, marker='o', color=color_S[base_S_color_i], label=f"SISA (S = {csv_name_info[0]})")
	else:
		ax[1].plot(x, y, z, marker='D', color='black', label=f"Batch K=") # K m thay thế thành batch size m chọn với AR Face Database

	if x.max() > max_nb_requests:
		max_nb_requests = x.max()
	if x.min() < min_nb_requests:
		min_nb_requests = x.min()
	if y.max() > max_time:
		max_time = y.max()
	if y.min() < min_time:
		min_time = y.min()
	if z.max() > max_acc:
		max_acc = z.max()
	if z.min() < min_acc:
		min_acc = z.min()

	base_S_color_i += 1

for csv_a_part_of_S in csv_files_lst_orl_part_of_S:
	csv_name_info = np.array(re.findall('\d+', file_dir), dtype=int)
	df = pd.read_csv(csv_S)
	x = df['nb_requests'].values
	y = df['retraining_time'].values
	z = df['accuracy'].values * 100
	ax[1].plot(x, y, z, marker='o', color=color_part_of_S[base_a_part_of_S_color_i], label=f"SISA (S = 1/{csv_name_info[0]})")
	
	if x.max() > max_nb_requests:
		max_nb_requests = x.max()
	if x.min() < min_nb_requests:
		min_nb_requests = x.min()
	if y.max() > max_time:
		max_time = y.max()
	if y.min() < min_time:
		min_time = y.min()
	if z.max() > max_acc:
		max_acc = z.max()
	if z.min() < min_acc:
		min_acc = z.min()

	base_a_part_of_S_color_i += 1


max_nb_requests, min_nb_requests, max_time, min_time, max_acc, min_acc = int(round(max_nb_requests,0)) + 1, m.floor(min_nb_requests) - 1,  
																		int(round(max_time,0)) + 1, m.floor(min_time) - 1, 
																		int(round(max_acc,0)) + 1, int(m.floor(min_acc)) - 1
step_nb_requests, step_time, step_acc = max_nb_requests - min_nb_requests, max_time - min_time, max_acc - min_acc

if step_time == 0 and step_acc == 0:
	step_nb_requests, step_time, step_acc = 1, 1, 1
else:
	step_nb_requests, step_time, step_acc = step_nb_requests // 4, step_time // 4, step_acc // 4

ax[1].set_xticks(np.arange(min_nb_requests, max_nb_requests, step_nb_requests))
ax[1].set_yticks(np.arange(min_time, max_time, step_time))
ax[1].set_zticks(np.arange(min_acc, max_acc, step_acc))

ax[1].set_xlabel('Số yêu cầu', fontsize=15, **csfont)
ax[1].set_ylabel('Thời gian phân tích (s)', fontsize=15, **csfont)
ax[1].set_zlabel('Độ chính xác (%)', fontsize=15, **csfont)
ax[1].set_title('ORL (theo số phân đoạn)', fontsize=20, fontweight="bold", **csfont)

plt.legend(bbox_to_anchor=(0.75, -0.15), fontsize=15, ncol = 9, prop={'family': 'serif'})
plt.savefig('face_plot_base_on_shards.png')
plt.show()
