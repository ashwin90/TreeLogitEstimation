import pandas as pd
import numpy as np
from itertools import product
import os
from collections import defaultdict
pd.set_option('precision', 1)
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from IPython import embed

simulation_output = 'Simulation/output/'
sushi_output = 'Sushi/output/'
num_sim_exps = 5
num_sushi_exps = 400
num_sushi_individuals = 5000
plot_dir = 'Simulation/plots'


def compute_agd_vs_amm_metrics():
	negloggap_info = None
	for fname in range(1, num_sim_exps + 1):
		# our_method metrics_one_instance
		output_file_name = 'Simulation/output/output' + str(fname) + '.txt'
		# output_file_name = amm_folder_name + fname
		# if not os.path.exists(output_file_name):
		#	continue
		metrics_one_instance = pd.read_csv(output_file_name)
		amm_metrics = metrics_one_instance[metrics_one_instance['method'] == 'fullback_mm']
		negloggap_info_id = amm_metrics[['depth', 'degree', 'lambda_lb']].copy()
		negloggap_info_id[['amm_time', 'amm_gap']] = amm_metrics[['time', 'll_diff']]
		# gd_output_file_name = agd_folder_name + fname
		# metrics_one_instance = pd.read_csv(gd_output_file_name)
		agd_metrics = metrics_one_instance[metrics_one_instance['method'] == 'agd_mu_delta']
		negloggap_info_id['agd_time'] = agd_metrics['time'].values
		negloggap_info_id['agd_gap'] = agd_metrics['ll_diff'].values
		negloggap_info_id['amm_better_than_pgd'] = negloggap_info_id['amm_gap'] < negloggap_info_id['agd_gap']
		if negloggap_info is None:
			negloggap_info = negloggap_info_id
		else:
			negloggap_info = pd.concat([negloggap_info, negloggap_info_id], ignore_index=True, sort=False)

	# compute the metrics_one_instance
	#print(negloggap_info.head(20))
	print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).mean()[['amm_gap', 'agd_gap', 'amm_better']])
	#print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).apply(lambda df: ttest_rel(df['amm_gap'], df['agd_gap'])[1]))


def compute_simulation_metrics():
	negloggap_info = None
	for fname in range(1, num_sim_exps + 1):
		# our_method metrics_one_instance
		output_file_name = simulation_output + 'output' + str(fname) + '.txt'
		# output_file_name = amm_folder_name + fname
		# if not os.path.exists(output_file_name):
		#	continue
		metrics_one_instance = pd.read_csv(output_file_name)
		# amm metrics
		amm_metrics = metrics_one_instance[metrics_one_instance['method'] == 'fullback_mm']
		negloggap_info_id = amm_metrics[['depth', 'degree', 'lambda_lb']].copy()
		negloggap_info_id[['amm_time', 'amm_gap']] = amm_metrics[['time', 'll_diff']]
		# pgd metrics
		pgd_metrics = metrics_one_instance[metrics_one_instance['method'] == 'gdmu']
		negloggap_info_id['pgd_time'] = pgd_metrics['time'].values
		negloggap_info_id['pgd_gap'] = pgd_metrics['ll_diff'].values
		negloggap_info_id['amm_better_than_pgd'] = negloggap_info_id['amm_gap'] < negloggap_info_id['pgd_gap']
		# knitro metrics (no rows for degree 8 so need to merge)
		knitro_metrics = metrics_one_instance[metrics_one_instance['method'] == 'knitro'][["depth", "degree", "lambda_lb", "time", "ll_diff"]]
		negloggap_info_id = pd.merge(negloggap_info_id, knitro_metrics, how="left", on=["depth", "degree", "lambda_lb"])
		negloggap_info_id.rename({"time": "knitro_time", "ll_diff": "knitro_gap"}, axis="columns", inplace=True)
		#negloggap_info_id['knitro_time'] = knitro_metrics['time'].values
		#negloggap_info_id['knitro_gap'] = knitro_metrics['ll_diff'].values
		negloggap_info_id['amm_better_than_knitro'] = negloggap_info_id['amm_gap'] < negloggap_info_id['knitro_gap']
		# concatenate metrics
		if negloggap_info is None:
			negloggap_info = negloggap_info_id
		else:
			negloggap_info = pd.concat([negloggap_info, negloggap_info_id], ignore_index=True, sort=False)

	# compute average gaps
	print('Printing average negloggaps')
	print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).mean()[['amm_gap', 'pgd_gap', 'knitro_gap']])
	# compute fraction better instances
	print('Printing fraction of instances where a-mm is better')
	print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).mean()[['amm_better_than_pgd', 'amm_better_than_knitro']])
	# compute p-value of t-test significance
	print('Printing p-value for difference between a-mm and pgd')
	print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).apply(lambda df: ttest_rel(df['amm_gap'], df['pgd_gap'])[1]))
	print('Printing p-value for difference between a-mm and knitro')
	print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).apply(lambda df: ttest_rel(df['amm_gap'], df['knitro_gap'])[1]))


def compute_sushi_metrics():
	warm_start_metrics_train = defaultdict(list)
	warm_start_metrics_test = defaultdict(list)
	zero_one_start_metrics_train = defaultdict(list)
	zero_one_start_metrics_test = defaultdict(list)
	for fname in range(1, num_sushi_exps + 1):
		# first read in mnl metrics
		output_file_name = sushi_output + 'sushi_mnl' + str(fname) + '.txt'
		with open(output_file_name, 'r') as fhandle:
			mnl_metrics_instance = fhandle.readline().strip().split(',')
		# nested logit with 0/1 start
		output_file_name = sushi_output + 'sushi_nl_01start' + str(fname) + '.txt'
		with open(output_file_name, 'r') as fhandle:
			nl_metrics_01start_instance = fhandle.readline().strip().split(',')
		# nested logit with warm start
		output_file_name = sushi_output + 'sushi_nl_warmstart' + str(fname) + '.txt'
		with open(output_file_name, 'r') as fhandle:
			nl_metrics_warmstart_instance = fhandle.readline().strip().split(',')

		# add train metrics
		zero_one_start_metrics_train['amm'].append(float(mnl_metrics_instance[0]) - float(nl_metrics_01start_instance[0]))
		zero_one_start_metrics_train['knitro'].append(float(mnl_metrics_instance[1]) - float(nl_metrics_01start_instance[1]))
		zero_one_start_metrics_train['pgd'].append(float(mnl_metrics_instance[2]) - float(nl_metrics_01start_instance[2]))
		warm_start_metrics_train['amm'].append(float(mnl_metrics_instance[0]) - float(nl_metrics_warmstart_instance[0]))
		warm_start_metrics_train['knitro'].append(float(mnl_metrics_instance[1]) - float(nl_metrics_warmstart_instance[1]))
		warm_start_metrics_train['pgd'].append(float(mnl_metrics_instance[2]) - float(nl_metrics_warmstart_instance[2]))

		# add test metrics
		zero_one_start_metrics_test['amm'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_01start_instance[3]))
		zero_one_start_metrics_test['knitro'].append(float(mnl_metrics_instance[4]) - float(nl_metrics_01start_instance[4]))
		zero_one_start_metrics_test['pgd'].append(float(mnl_metrics_instance[5]) - float(nl_metrics_01start_instance[5]))
		warm_start_metrics_test['amm'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_warmstart_instance[3]))
		warm_start_metrics_test['knitro'].append(float(mnl_metrics_instance[4]) - float(nl_metrics_warmstart_instance[4]))
		warm_start_metrics_test['pgd'].append(float(mnl_metrics_instance[5]) - float(nl_metrics_warmstart_instance[5]))

	# compute metrics for 0/1 start
	print('Printing train metrics for 0/1 start')
	print([(method, num_sushi_individuals*np.mean(zero_one_start_metrics_train[method])) for method in zero_one_start_metrics_train])
	print('Printing test metrics for 0/1 start')
	print([(method, num_sushi_individuals*np.mean(zero_one_start_metrics_test[method])) for method in zero_one_start_metrics_test])
	# compute metrics for warm start
	print('Printing train metrics for warm start')
	print([(method, num_sushi_individuals*np.mean(warm_start_metrics_train[method])) for method in warm_start_metrics_train])
	print('Printing test metrics for warm start')
	print([(method, num_sushi_individuals*np.mean(warm_start_metrics_test[method])) for method in warm_start_metrics_test])

	# compute p-value of t-test significance
	#print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).apply(lambda df: ttest_rel(df['amm_gap'], df['pgd_gap'])[1]))
	#print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).apply(lambda df: ttest_rel(df['amm_gap'], df['knitro_gap'])[1]))


def plot_negloggaps_vs_iters(degree, depth, lambda_lb):
	pgd_file_name = '{3}/pgd_{0}_{1}_{2}.err'.format(depth, degree, lambda_lb, plot_dir)
	amm_file_name = '{3}/amm_{0}_{1}_{2}.err'.format(depth, degree, lambda_lb, plot_dir)
	agd_file_name = '{3}/agd_{0}_{1}_{2}.err'.format(depth, degree, lambda_lb, plot_dir)
	with open(amm_file_name, 'r') as fhandle:
		amm_stats = fhandle.readlines()
	with open(pgd_file_name, 'r') as fhandle:
		pgd_stats = fhandle.readlines()
	with open(agd_file_name, 'r') as fhandle:
		agd_stats = fhandle.readlines()

	num_iters = 200
	true_neglog = float(amm_stats[0].split(":")[1])
	pgd_negloggaps = [float(line.split()[-4]) - true_neglog for line in pgd_stats[1:num_iters+2]]
	amm_negloggaps = [float(line.split()[-4]) - true_neglog for line in amm_stats[1:num_iters+2]]
	agd_negloggaps = [float(line.split()[-4]) - true_neglog for line in agd_stats[1:num_iters+2]]
	plt.yscale('log')
	plt.plot(range(0, num_iters + 1), agd_negloggaps, label='A-GD')
	plt.plot(range(0, num_iters + 1), amm_negloggaps, label='A-MM')
	plt.plot(range(0, num_iters + 1), pgd_negloggaps, label='PGD')
	plt.xlabel('Iteration number')
	plt.ylabel('NegLogGap')
	plt.legend(loc='best')
	plt.savefig('{3}/negloggaps_vs_iters_all_{0}_{1}_{2}.pdf'.format(degree, depth, lambda_lb, plot_dir))


def plot_negloggaps_vs_iters_single_file():
	file_name = 'plots/agd_vs_amm.err'
	with open(file_name, 'r') as fhandle:
		complete_stats = fhandle.readlines()
	run_index = 1
	num_agd_iters = 300
	num_amm_iters = 400
	lines_per_run = num_agd_iters + num_amm_iters + 3
	for degree in [6, 7]:
		true_neglog = float(complete_stats[lines_per_run*(run_index-1)].split(":")[1])
		mm_start_line = lines_per_run*(run_index - 1) + 1
		mm_end_line = mm_start_line + num_agd_iters + 1
		print(mm_start_line, mm_end_line)
		mm_negloggaps = [float(line.split()[-4]) - true_neglog for line in complete_stats[mm_start_line:mm_end_line]]
		agd_start_line = lines_per_run*(run_index - 1) + num_amm_iters + 2
		agd_end_line = agd_start_line + num_agd_iters + 1
		print(agd_start_line, agd_end_line)
		agd_negloggaps = [float(line.split()[-4]) - true_neglog for line in complete_stats[agd_start_line:agd_end_line]]
		plt.clf()
		plt.plot(range(0, num_agd_iters+1), agd_negloggaps, label='A-GD')
		plt.plot(range(0, num_agd_iters+1), mm_negloggaps, label='A-MM')
		plt.yscale('log')
		plt.xlabel('Iteration number')
		plt.ylabel('NegLogGap')
		plt.legend(loc='best')
		plt.savefig('plots/negloggaps_agd_vs_amm_degree={0}.pdf'.format(degree))
		run_index += 1


if __name__ == "__main__":
	plot_negloggaps_vs_iters(8, 5, 0.01)
	#plot_negloggaps_vs_iters_single_file()
	#compute_simulation_metrics()
	#compute_sushi_metrics()