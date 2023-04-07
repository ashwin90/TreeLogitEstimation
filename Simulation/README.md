This folder contains the code and data files to replicate the results in Section 4.1 of the paper.

The data files are located under the `output/mfile` folder. There is a folder for each ground-truth setting in Figure 3, labeled in the format `height`-`degree`-$\lambda_{\text{lower}} $. Within each folder, each `.mat` file contains the input sales transaction data for a different problem instance under that ground-truth.

Follow the steps below to generate the simulation metrics:

1. Run the matlab file `run_scenarios.m` for each problem instance specifying the instance number (1, 2, ..., 100) as an argument. This will generate 100 files of the form `ouput<instance_num>.txt` inside the `output` folder.

2. Run the python file `analyze_results.py` by calling the `compute_simulation_metrics()` to generate the table shown in Figure 3.
