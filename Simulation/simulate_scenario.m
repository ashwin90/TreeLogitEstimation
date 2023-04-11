function simulate_scenario(output_file, num_levels, degree, n_offersets, max_sales, cutoff, lambda_lb, lambda_ub, mean_util_mult, n_iters, file_open_mode,number)
    % simulation parameters
    % num_levels = num_levels;
    % degree = 10;
    % n_offersets = 60;
    % max_sales = 100;
    % cutoff = 0.1; % controls offerset size
    % lambda_lb = 0.01;
    % lambda_ub = 0.1;
    % mean_util_mult = 1;


    num_prods = power(degree, num_levels);
    adj_mats = {ones(degree, 1)};
    for level=1:num_levels-1
        nrows = power(degree, level+1);
        ncols = power(degree, level);
        m = adj_mats{level};
        % the code below constructs a block diagonal matrix
        mr = repmat(m, 1, degree);
        mc = mat2cell(mr, size(m,1), repmat(size(m,2),1,degree));

        adj_mats{level+1} = blkdiag(mc{:});
    end
    % NOTE: the levels are numbered such that the leaf nodes are level 1 and
    % root node is level d
    adj_mats = fliplr(adj_mats);

    metrics = {'ll_diff', 'rel_mu_err', 'rel_lambda_err'};
    metricsTypes = {'double', 'double', 'double'};
    addln_metrics = {'time', 'infeasible_lambdas', 'infeasible_type'};
    addln_metricsTypes = {'double', 'logical','double'};
    %methods = {'fullback_mm', 'agd_mu_delta'};
    methods = {'fullback_mm','gdmu', 'knitro'};
    % create a MATLAB table of the metrics
    varNames = {'depth', 'degree', 'lambda_lb', 'lambda_ub', 'method'};
    varTypes = {'unit32', 'unit32', 'double', 'double', 'string'};
    varNames = {varNames{:}, metrics{:}, addln_metrics{:}};
    varTypes = {varTypes{:}, metricsTypes{:}, addln_metricsTypes{:}};
    sz = [n_iters, size(varNames, 2)];
%   output_T = table('VariableTypes', varTypes, 'VariableNames', varNames);

    % create header string
    header_str = strjoin(varNames, ',');
    common_params_str = {num2str(num_levels), num2str(degree), num2str(lambda_lb), num2str(lambda_ub)};
%     common_params_str = arrayfun(@num2str, common_params, 'Uniform', false); %['%d', '%d', '%f'], 
    

%     fID = fopen('output.txt', 'w');
    if strcmp(file_open_mode, 'write')
        fID = fopen(output_file, 'w');
        fprintf(fID, strcat(header_str, '\n'));
    else
        fID = fopen(output_file, 'a');
    end
    
    % simulate purchase data
    NITERS = n_iters;
    for iters=1:NITERS
        % generate ground truth parameters
        filename = [num2str(num_levels),'-',num2str(degree),'-',num2str(lambda_lb)];
        % load data for problem instance
        	load(['output/mfile/',filename,'/',num2str(number),'.mat'])
            
            ground_truth_m.adj_mats = adj_mats;

        ground_truth_m.set_fwmm_params();
        sales = max_sales .* ground_truth_m.compute_proba(osets);

        fitted_models = cell(1, size(methods, 2));
        op_metrics = [];
        fvals = zeros(1, size(methods, 2));
        fittimes = zeros(1, size(methods, 2));
        true_nLL = -ground_truth_m.logLL(osets, sales);
        fprintf(2, 'Ground-truth nll:%f\n', true_nLL);
        for i=1:size(methods, 2)
            fitted_models{i} = NLdSIMUL(num_prods, num_levels, adj_mats, [], {});
            fitted_models{i}.set_fwmm_params();
            tic;
            fvals(i) = fitted_models{i}.fit(osets, sales, -Inf, 2*mean_util_mult, lambda_lb, methods{i});
            fittimes(i) = toc;
            % check feasibility of nest dissimilarity parameters
            tmp_model = fitted_models{i};
            infeasible_flag = 0;
            feasible_params = true;
            feasible_params = feasible_params && all(tmp_model.nest_dissim_params{1} > 0);
            
            if(~all(tmp_model.nest_dissim_params{1} > 0))
                infeasible_flag = 1;
            end 
            
            for level = 2:tmp_model.num_levels
                feasible_params = feasible_params && all(tmp_model.nest_dissim_params{level-1} <= 0.000001+tmp_model.adj_mats{level}*tmp_model.nest_dissim_params{level});                
                if(~all(tmp_model.nest_dissim_params{level-1} <= (0.000001+tmp_model.adj_mats{level}*tmp_model.nest_dissim_params{level})))
                    infeasible_flag = infeasible_flag + 10;
                end
            
            end
            
            op_metrics = compute_metrics(metrics, ground_truth_m, fitted_models{i}, osets, sales);
            addln_op_metrics = [fittimes(i), ~feasible_params, infeasible_flag];
            
            % convert all numeric arrays to strings
            
            op_metrics_str = arrayfun(@num2str, op_metrics, 'Uniform', false);
            addln_op_metrics_str = arrayfun(@num2str, addln_op_metrics, 'Uniform', false);
            
            output_str = strcat(strjoin(common_params_str, ','), ',', methods{i}, ',', strjoin(op_metrics_str, ','), ',', strjoin(addln_op_metrics_str, ','), '\n');
            fprintf(fID, output_str);
%             if i == 1
%                 op_metrics = compute_metrics(metrics, ground_truth_m, fitted_models{i}, osets, sales);
%                 op_metrics = [op_metrics, fittimes(i), ~feasible_params];
%             else
%                 op_metrics = [op_metrics, compute_metrics(metrics, ground_truth_m, fitted_models{i}, osets, sales)];
%                 op_metrics = [op_metrics, fittimes(i), ~feasible_params];
%             end

                
        end
%         output_str = strjoin(arrayfun(@num2str, op_metrics, 'Uniform', false), ',');
%         fprintf(fID, strcat(output_str, '\n'));
        % fprintf('%f,%f,%f,%f\n', fvals(1) - true_nLL, fvals(2) - true_nLL, fittimes(1), fittimes(2));
	mmpgd_mu=fitted_models{1}.mean_utils;
	pgd_mu=fitted_models{2}.mean_utils;
	mmpgd_lambda=fitted_models{1}.nest_dissim_params;
	pgd_lambda=fitted_models{2}.nest_dissim_params;


	%save(['/scratch/xz2197/gdmu_store/parameters/',filename,'/',num2str(number),'.mat'],'mmpgd_mu', 'pgd_mu', 'mmpgd_lambda', 'pgd_lambda');
	%save(['/scratch/xz2197/gdmu_store/parameters/',filename,'/',num2str(number),'.mat'],'fitted_models')
    end
    fclose(fID);
end
