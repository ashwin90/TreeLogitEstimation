This folder contains the code and data files to replicate the results in Section 4.2 of the paper.

The data files and the tree structure are located under the `data` folder. In particular, the `data/data_0405_07` subfolder contains 400 `.mat` files, one for each instance. The `data/final_tree_0128` contains the `node.mat` file which encodes the tree structure shown in Appendix F of the paper.

Follow the steps below to generate the metrics reported in Figure 4:

1. To fit the MNL models, run the matlab file `Training_final_MNL(instance_id)` for each problem instance specifying the instance number, instance_id = 1, 2, ..., 400 as an argument. This will generate 400 files of the form `sushi_mnl<instance_num>.txt` inside the `output` folder.

2. To fit the tree logit models with 0/1 initialization, run the matlab file `Training_final_NL(instance_id, false)` for each problem instance specifying the instance number, instance_id = 1, 2, ..., 400 as an argument. This will generate 400 files of the form `sushi_nl_01start<instance_num>.txt` inside the `output` folder. For warm start initialization, run the matlab file `Training_final_NL(instance_id, true)` for each problem instance specifying the instance number, instance_id = 1, 2, ..., 400 as an argument. This will generate 400 files of the form `sushi_nl_warmstart<instance_num>.txt` inside the `output` folder [NOTE for warm start, step 1 above must have finished since the code reads the saved MNL parameters.]


3. Run the following command

> python analyze_results.py sushi
