This folder contains the code and data files to replicate the results in Section 4.1 of the paper.

The data files are located under the `output/mfile` folder. There is a folder for each ground-truth setting in Figure 3, labeled in the format `height`-`degree`-$\lambda_{\mathrm{lower}}$. Within each folder, each `.mat` file contains the input sales transaction data for a different problem instance under that ground-truth.

Follow the steps below to generate the metrics shown in Figure 3:

1. Run the matlab file `run_scenarios(instance_id)` for each problem instance specifying the instance number, instance_id = 1, 2, ..., 100 as an argument. This will generate 100 files of the form `output<instance_num>.txt` inside the `output` folder.

2. Run the following command 

> python analyze_results.py simulation

To compute the numbers in Figure EC.3, uncomment line 33 and comment line 34 in the file `simulate_scenario.m`. Then, repeat step 1 above and run the following command:

> python analyze_results.py simulation-appendix
