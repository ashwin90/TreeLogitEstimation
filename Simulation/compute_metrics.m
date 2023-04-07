function op_metrics = compute_metrics(metrics, gt_model, fit_model, osets, sales)
    op_metrics = zeros(1, size(metrics,2));
    count = 1;
    for m=metrics
        if strcmp(m, 'll_diff')
            op_metrics(count) = gt_model.logLL(osets, sales) - fit_model.logLL(osets, sales);
        elseif strcmp(m, 'rel_mu_err')
            op_metrics(count) = norm(gt_model.mean_utils - fit_model.mean_utils)/norm(gt_model.mean_utils);
        elseif strcmp(m, 'rel_lambda_err')
            %create an array of all lambdas
            gt_lambdas = zeros(sum(gt_model.num_nodes_per_level)-gt_model.num_prods, 1);
            fit_lambdas = zeros(sum(fit_model.num_nodes_per_level)-fit_model.num_prods, 1);
            cum_sum_nodes_per_level = cumsum(gt_model.num_nodes_per_level(2:end));
            cum_sum_nodes_per_level = [0; cum_sum_nodes_per_level];
            for level = 1:gt_model.num_levels
                gt_lambdas(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))= gt_model.nest_dissim_params{level};
                fit_lambdas(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1)) = fit_model.nest_dissim_params{level};
            end
            op_metrics(count) = norm(gt_lambdas - fit_lambdas)/norm(gt_lambdas);
        end
        count = count + 1;
    end
end