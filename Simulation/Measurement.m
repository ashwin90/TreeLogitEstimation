function Measurement(output_file, num_levels, degree, n_offersets, max_sales, cutoff, lambda_lb, lambda_ub, mean_util_mult, n_iters, file_open_mode,number)

%top_20 = [1:20];
   filename = [num2str(num_levels),'-',num2str(degree),'-',num2str(lambda_lb)];
   
   load(['output/mfile/',filename,'/',num2str(number),'.mat'])
            
        ground_truth_m.adj_mats = adj_mats;

        ground_truth_m.set_fwmm_params();
        sales = max_sales .* ground_truth_m.compute_proba(osets);

        
   load(['/scratch/xz2197/gdmu_store/parameters/',filename,'/',num2str(number),'.mat']);
   methods = {'fullback_mm','gdmu'};
   

    output_file = ['output_measurement/',num2str(k),'.txt'];
    
    rmse = zeros(1, size(methods, 2));
    mape = zeros(1, size(methods, 2));

    chi_soft = zeros(1, size(methods, 2));
    chi_hard = zeros(1, size(methods, 2));
    mape_soft = zeros(1, size(methods, 2));
    mape_hard = zeros(1, size(methods, 2));
    
    
   for methodid =1:size(methods, 2)
       
       rmse(methodid) = fitted_models{methodid}.rmse(testing_offerset, testing_sales);
       mape(methodid) = fitted_models{methodid}.abss(testing_offerset, testing_sales);
       
       [chi_soft(methodid),chi_hard(methodid),mape_soft(methodid),mape_hard(methodid)] = fitted_models{methodid}.Ni(testing_offerset, testing_sales);
       
   end

    
     fID = fopen(output_file, 'w');
    %output_str = test_fvals;
            fprintf(fID,'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n',...
                rmse(1),rmse(2),mape(1),mape(2),...
                chi_soft(1),chi_soft(2),...
                chi_hard(1),chi_hard(2),...
                mape_soft(1),mape_soft(2),...
                mape_hard(1),mape_hard(2));
                     
        
        fclose(fID);  
    
    %Training_final(k);
    fprintf('%f\n', k);

end 
    
   