function run_scenarios(number)
rng(number);
% script to run different scenarios to compare fmincon and fw_mm algos
% for fitting tree logit models

num_levels = 5;
degree = 3;
n_offersets = 60;
max_sales = 100;
cutoff = 0.1; % controls offerset size
lambda_lb = 0.5;
lambda_ub = 1;
mean_util_mult = 1;
n_iters = 1;
fmincon_MAX_ITERS = 50000;

% key problem params
% num_levels: depth of the logit tree
% degree: out-degree of each node in the tree (except the leaf nodes)
% lambda_lb: lower bound on the value of the nest dissim param

% the depth and degree params control the size of the problem
% the value of lambda_lb controls the hardness of estimation
% the goal is to run a horse race of methods as a function of the size and
% lambda_lb


depth_vals = [2,3];
degree_vals = [3,4,5];
lambda_lb_vals = [0.01, 0.1, 0.5];
[X, Y, Z] = meshgrid(depth_vals, degree_vals, lambda_lb_vals);
param_comb_vals = [X(:), Y(:), Z(:)];
counter = 1;
for i=1:size(param_comb_vals, 1)
    param = param_comb_vals(i, :);
    num_levels = int32(param(1));
    degree = int32(param(2));
    lambda_lb = param(3);
    if counter == 1
        mode = 'write';
    else
        mode = 'append';
    end
    %simulate_scenario('output.txt', num_levels, degree, n_offersets, max_sales, cutoff, lambda_lb, lambda_ub, mean_util_mult, n_iters, mode);
    simulate_scenario(['output/output',num2str(number),'.txt'], num_levels, degree, n_offersets, max_sales, cutoff, lambda_lb, lambda_ub, mean_util_mult, n_iters, mode,number);
    counter = counter + 1;
end
end
