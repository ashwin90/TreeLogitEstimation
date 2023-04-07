function Training_final_MNL(number)

    num_levels = 5;
    degree = 2;
    n_iters = 1;
    init_mnl = true;
    
    num_prods = 101;
    
    % tree structure
    load(['data/final_tree_0128/node.mat']);
    output_file = ['output/sushi_mnl',num2str(number),'.txt'];
    %mode = 'write';
    %mode = 'append';

    % load transaction data
    load(['data/data_0405_07/',num2str(number),'.mat']);
    
    training_offerset = training_offerset(:,index_order);
    testing_offerset = testing_offerset(:,index_order);
    training_sales = training_sales(:,index_order)/5000;
    
    testing_sales = testing_sales(:,index_order)/5000;
    methods = {'fullback_mnl','knitro_mnl','gdmu_mnl'};
    fitted_models = cell(1, size(methods, 2));
    %op_metrics = [];
    train_fvals = zeros(1, size(methods, 2));
    test_fvals = zeros(1, size(methods, 2));
    %rmse = zeros(1, size(methods, 2));
    %abss = zeros(1, size(methods, 2));
    fittimes = zeros(1, size(methods, 2));
        
    fitted_models{1} = NLdSIMUL(num_prods, 5, adj_mats, [], {});
    fitted_models{2} = NLdSIMUL(num_prods, 5, adj_mats, [], {});
    fitted_models{3} = NLdSIMUL(num_prods, 5, adj_mats, [], {});
    %fitted_models{4} = NLdSIMUL(num_prods, 5, adj_mats, [], {});
    
    tic;
     [train_fvals(1),mu, lambda]= fitted_models{1}.train(training_offerset, training_sales, -Inf, 2, 0.01, methods(1), number, init_mnl);
     fittimes(1) = toc;
     ground_truth_m_1 = NLdSIMUL(num_prods, 5, adj_mats, mu, lambda);
    test_fvals(1) = -ground_truth_m_1.logLL(testing_offerset, testing_sales);
    %rmse(1) = ground_truth_m_1.rmse(testing_offerset, testing_sales);
    %abss(1) = ground_truth_m_1.abss(testing_offerset, testing_sales);
    
    tic;
     [train_fvals(2),mu, lambda]= fitted_models{2}.train(training_offerset, training_sales, -Inf, 2, 0.01, methods(2), number, init_mnl);
     fittimes(2) = toc;
     ground_truth_m_2 = NLdSIMUL(num_prods, 5, adj_mats, mu, lambda);
    test_fvals(2) = -ground_truth_m_2.logLL(testing_offerset, testing_sales);
    %rmse(2) = ground_truth_m_2.rmse(testing_offerset, testing_sales);
    %abss(2) = ground_truth_m_2.abss(testing_offerset, testing_sales);
        
    tic;
     [train_fvals(3),mu, lambda]= fitted_models{3}.train(training_offerset, training_sales, -Inf, 2, 0.01, methods(3),number, init_mnl);
     fittimes(3) = toc;
     ground_truth_m_3 = NLdSIMUL(num_prods, 5, adj_mats, mu, lambda);
    test_fvals(3) = -ground_truth_m_3.logLL(testing_offerset, testing_sales);
    %rmse(3) = ground_truth_m_3.rmse(testing_offerset, testing_sales);
    %abss(3) = ground_truth_m_3.abss(testing_offerset, testing_sales);
    
%     tic;
%      [train_fvals(4),mu, lambda]= fitted_models{4}.train(training_offerset, training_sales, -Inf, 2, 0.01, methods(4));
%      fittimes(4) = toc;
%      ground_truth_m_4 = NLdSIMUL(num_prods, 5, adj_mats, mu, lambda);
%     test_fvals(4) = -ground_truth_m_4.logLL(testing_offerset, testing_sales);
    %rmse(4) = ground_truth_m_4.rmse(testing_offerset, testing_sales);
    %abss(4) = ground_truth_m_4.abss(testing_offerset, testing_sales);

     save(['ground_truth_mnl/',num2str(number),'.mat'],'ground_truth_m_1','ground_truth_m_2','ground_truth_m_3');
    
        fID = fopen(output_file, 'w');
    %output_str = test_fvals;
            fprintf(fID,'%f,%f,%f,%f,%f,%f,%f,%f,%f\n',train_fvals(1),train_fvals(2),train_fvals(3),...
                test_fvals(1),test_fvals(2),test_fvals(3),fittimes(1),fittimes(2),fittimes(3));
            
            
        
        fclose(fID);    
       
end