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
    
    % create a MATLAB table of the metrics
    

    if strcmp(file_open_mode, 'write')
        fID = fopen(output_file, 'w');
        fprintf(fID, strcat(header_str, '\n'));
    else
        fID = fopen(output_file, 'a');
    end
    
    filename = [num2str(num_levels),'-',num2str(degree),'-',num2str(lambda_lb)];
    load(['output/mfile/',filename,'/',num2str(number),'.mat']);
    
    
    check_item = sum(osets);
    
    
    
    % simulate purchase data
    NITERS = num_prods/degree-1;
    
    for iters=0:NITERS
        current_sum = sum(check_item(degree*iters+1:degree*(iters+1)));
        

        if(current_sum < 60)
            disp('Error');
        end
  
        
    end

    
    %fclose(fID);
end