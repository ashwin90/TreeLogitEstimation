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
    addln_metrics = {'time', 'infeasible_lambdas'};
    addln_metricsTypes = {'double', 'logical'};
    methods = {'fw_mm','fw_mm'};
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
    
%     % create header string
%     header_str = '';
%     for i=1:size(methods, 2)
%         if i==1
%             header_str = strjoin(strcat(methods(i), '_', metrics), ',');
%             header_str = strcat(header_str, ',', strcat(methods{i}, '_time'));
%             header_str = strcat(header_str, ',', strcat(methods{i}, '_infeasible_lambdas'));
%         else
%             header_str = strcat(header_str, ',', strjoin(strcat(methods(i), '_', metrics), ','));
%             header_str = strcat(header_str, ',', strcat(methods{i}, '_time'));
%             header_str = strcat(header_str, ',', strcat(methods{i}, '_infeasible_lambdas'));
%         end
%     end

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
        mean_utils = mean_util_mult*rand(1, num_prods);
        %mean_utils = mean_util_mult*0.5*ones(1, num_prods);
        mean_utils = 1 + mean_utils - min(mean_utils);
        % assign a nest dissim param to each non-leaf node
        nest_dissim_params = {[1]};
        for level=1:num_levels-1
            if(level==1)
                nest_dissim_params{level} = lambda_lb + (lambda_ub - lambda_lb)*rand(power(degree,num_levels-level), 1);
                %nest_dissim_params{level} = lambda_lb + (lambda_ub - lambda_lb)*0.5*ones(power(degree,num_levels-level), 1);
            else
                % determine the lower bound for the nest dissim parameters
                child_lambdas = max(bsxfun(@times, adj_mats{level}, nest_dissim_params{level-1}))';
                nest_dissim_params{level} = child_lambdas + (lambda_ub - child_lambdas).*rand(power(degree,num_levels-level), 1);
                %nest_dissim_params{level} = child_lambdas + (lambda_ub - child_lambdas).*0.5*ones(power(degree,num_levels-level), 1);
            end
        end
        nest_dissim_params{num_levels} = [1];

        ground_truth_m = NLdSIMUL(num_prods, num_levels, adj_mats, mean_utils, nest_dissim_params);
        ground_truth_m.set_fwmm_params();
        osets = [rand(n_offersets,num_prods) > cutoff];
        %osets = [ones(n_offersets,num_prods) > cutoff];
        if(any(sum(osets, 2) == 0))
            continue;
        end
        %sales = max_sales .* ground_truth_m.compute_proba(osets);

        %fitted_models = cell(1, size(methods, 2));
        %op_metrics = [];
        %fvals = zeros(1, size(methods, 2));
        %fittimes = zeros(1, size(methods, 2));
        
        % save
        %save('pqfile.mat','ground_truth_m','osets')
        filename = [num2str(num_levels),'-',num2str(degree),'-',num2str(lambda_lb)];
        directoryname = ['output/mfile_small/',filename];
        mkdir(directoryname);
        ground_truth_m.adj_mats={};
        save(['output/mfile_small/',filename,'/',num2str(number),'.mat'],'ground_truth_m','osets')
        
    end
    fclose(fID);
end