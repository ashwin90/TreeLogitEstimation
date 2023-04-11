import pandas as pd
import numpy as np
from itertools import product
import os
from collections import defaultdict
pd.set_option('precision', 1)
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from IPython import embed
import sys

simulation_output = 'Simulation/output/'
sushi_output = 'Sushi/output/'
num_sim_exps = 100
num_sushi_exps = 400
num_sushi_individuals = 5000
plot_dir = 'Simulation/plots'


def compute_agd_vs_amm_metrics():
	negloggap_info = None
	for fname in range(1, num_sim_exps + 1):
		output_file_name = simulation_output + 'output' + str(fname) + '.txt'
		# if not os.path.exists(output_file_name):
		#	continue
		metrics_one_instance = pd.read_csv(output_file_name)
		# amm metrics
		amm_metrics = metrics_one_instance[metrics_one_instance['method'] == 'fullback_mm']
		negloggap_info_id = amm_metrics[['depth', 'degree', 'lambda_lb']].copy()
		negloggap_info_id[['amm_time', 'amm_gap']] = amm_metrics[['time', 'll_diff']]
		# agd metrics
		agd_metrics = metrics_one_instance[metrics_one_instance['method'] == 'agd_mu_delta']
		negloggap_info_id['agd_time'] = agd_metrics['time'].values
		negloggap_info_id['agd_gap'] = agd_metrics['ll_diff'].values
		negloggap_info_id['amm_better_than_pgd'] = negloggap_info_id['amm_gap'] < negloggap_info_id['agd_gap']
		if negloggap_info is None:
			negloggap_info = negloggap_info_id
		else:
			negloggap_info = pd.concat([negloggap_info, negloggap_info_id], ignore_index=True, sort=False)

	# compute average gaps and fraction better instances
	print('Printing average negloggaps and fraction of instances where a-mm is better than a-gd')
	print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).mean()[['amm_gap', 'agd_gap', 'amm_better']])
	# compute p-value of t-test significance
	print('Printing p-value for difference between a-mm and a-gd')
	print(negloggap_info.groupby(['degree', 'depth', 'lambda_lb']).apply(lambda df: ttest_rel(df['amm_gap'], df['agd_gap'])[1]))


def compute_simulation_metrics():
	negloggap_info = None
	for fname in range(1, num_sim_exps + 1):
		output_file_name = simulation_output + 'output' + str(fname) + '.txt'
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
		# knitro metrics (no rows for large instances so need to merge)
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
		zero_one_start_metrics_train['knitro'].append(float(mnl_metrics_instance[0]) - float(nl_metrics_01start_instance[1]))
		zero_one_start_metrics_train['pgd'].append(float(mnl_metrics_instance[0]) - float(nl_metrics_01start_instance[2]))
		warm_start_metrics_train['amm'].append(float(mnl_metrics_instance[0]) - float(nl_metrics_warmstart_instance[0]))
		warm_start_metrics_train['knitro'].append(float(mnl_metrics_instance[0]) - float(nl_metrics_warmstart_instance[1]))
		warm_start_metrics_train['pgd'].append(float(mnl_metrics_instance[0]) - float(nl_metrics_warmstart_instance[2]))

		# add test metrics
		zero_one_start_metrics_test['amm'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_01start_instance[3]))
		zero_one_start_metrics_test['knitro'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_01start_instance[4]))
		zero_one_start_metrics_test['pgd'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_01start_instance[5]))
		warm_start_metrics_test['amm'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_warmstart_instance[3]))
		warm_start_metrics_test['knitro'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_warmstart_instance[4]))
		warm_start_metrics_test['pgd'].append(float(mnl_metrics_instance[3]) - float(nl_metrics_warmstart_instance[5]))

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


if __name__ == "__main__":
	option = sys.argv[1]
	if option == 'sushi':
		compute_sushi_metrics()
	elif option == 'simulation':
		compute_simulation_metrics()
	else:
		compute_agd_vs_amm_metrics()
