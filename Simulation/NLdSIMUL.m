    classdef NLdSIMUL < handle
    %NLdSIMUL Simulator for a d-level nested logit model
    % The model is described as a tree with d levels
    % Each non-leaf node is associated with a nest dissimilarity parameter
    % Each leaf node is assocaited with a utility parameter
    %   Detailed explanation goes here
    
    properties
        num_prods = 1;
        num_levels = 1;
        adj_mats = []; % list of adjacency matrices describing tree structure
        num_nodes_per_level = [];
        % model parameters
        mean_utils = [];
        nest_dissim_params = []; % cell array storing paramteres layer-by-layer
        %fwmm parameters
        fwmm_niters = 0;
    end
    
    methods
        function obj = NLdSIMUL(num_prods, num_levels, adj_mats, mean_utils, nest_dissim_params)
            % adj_mats: cell array of length number levels
            % mean_utils: array of length num_prods
            % nest_dissim_params: cell array of length number of levels
            obj.num_prods = num_prods;
            obj.num_levels = num_levels;
            obj.adj_mats = adj_mats;
            obj.mean_utils = mean_utils;
            obj.nest_dissim_params = nest_dissim_params;
            obj.num_nodes_per_level = ones(obj.num_levels+1, 1);
            obj.num_nodes_per_level(1) = obj.num_prods;
            for level=2:obj.num_levels+1
                obj.num_nodes_per_level(level) = size(obj.adj_mats{level-1}, 2);
            end
        end
        
        function set_fwmm_params(obj, varargin)
            % set parameters for the fwmm method
            defaultNITERS = 200;
            validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0); % to check if the argument is postive scalar number
            p = inputParser;
            addOptional(p, 'n_iters', defaultNITERS, validScalarPosNum);
            % n_iters is the number of iterations for which MM algo is run
            parse(p, varargin{:});
            obj.fwmm_niters = p.Results.n_iters;
        end
        
        function set_fmincon_params(obj, varargin)
            % set parameters for the fmincon method
            defaultMaxIters = 0;
            validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (floor(x) == x); % to check if the argument is postive scalar number
            p = inputParser;
            addOptional(p, 'max_niters', defaultMaxIters, validScalarPosNum);
            % n_iters is the max number of iterations for fmincon
            parse(p, varargin{:});
            obj.fmincon_maxiters = p.Results.max_niters;
        end
                   
        function v_wts = feedforward(obj, offersets)
            % propagate the v_wts in the graph and output the v_wts at
            % each node
            v_weights = {bsxfun(@times, exp(obj.mean_utils), offersets)}; % initialize the v_wts at the leaf nodes
            for level = 1:obj.num_levels
                % compute the exponents
                expts = obj.adj_mats{level}*power(obj.nest_dissim_params{level}, -1);
                % normalize vs before raising them to the exponents
                prev_vs = v_weights{level};
                normalize_factor = prev_vs*obj.adj_mats{level};
                normalize_factor_expand = normalize_factor*obj.adj_mats{level}';
                prev_vs = prev_vs./(normalize_factor_expand  + (normalize_factor_expand  == 0)); % ensures that the exponents don't blow up
                
                % raise vs to exponents
                tmp_wts = bsxfun(@power, prev_vs, expts')*obj.adj_mats{level};
                % aggregate the vs to the next level, raise to lambda, and
                % un-normalize
                v_weights{level+1} = normalize_factor.*bsxfun(@power, tmp_wts, obj.nest_dissim_params{level}');
            end
            v_wts = v_weights;
        end

        function choice_probs = compute_proba(obj, offersets)
            % uses random walk to compute choice probabilities
            v_wts = obj.feedforward(offersets);
            num_offersets = size(offersets, 1);
            probs = ones(num_offersets, 1); % initialize the probability of choosing the root node to 1
            for level=obj.num_levels:-1:1
                repl_probs = probs*obj.adj_mats{level}'; % replicate probs of entering the nodes
                probs = obj.local_probs(level, v_wts).*repl_probs;
            end
            assert(all(round(abs(sum(probs, 2) - 1), 7) == 0));
            choice_probs = probs;
        end
	
	function [chi_soft,chi_hard,mape_soft,mape_hard] = Ni(obj, offersets, sales)
            % function to compute the abss
            prob_matrix = obj.compute_proba(offersets);
            
            Ni = sum(sales);
            Ni_soft = sum(prob_matrix);
            Ni_hard = sum(prob_matrix==max(prob_matrix,[],2));
            
            chi_soft = sum(((Ni-Ni_soft).^2)./(Ni_soft+0.5));
            chi_hard = sum(((Ni-Ni_hard).^2)./(Ni_hard+0.5));
            
            mape_soft = sum(abs(Ni-Ni_soft)./(Ni_soft+0.5));
            mape_hard = sum(abs(Ni-Ni_hard)./(Ni_hard+0.5));

        end

	function rmse_value = rmse(obj, offersets, sales)
            % function to compute the RMSE
            prob_matrix = obj.compute_proba(offersets);
            
            value = 0;
            total_row = size(offersets,1);
            for row = 1:total_row
                value = value + ((norm(sales(row,:)-prob_matrix(row,:)))^2)/sum(offersets(row,:));
                
                %%%%%sum(offersets(row,:))
                
            end
            rmse_value = sqrt(value/total_row);
        end
        
        function map_value = abss(obj, offersets, sales)
            % function to compute the abss
            prob_matrix = obj.compute_proba(offersets);
            
            value = 0;
            total_row = size(offersets,1);
            for row = 1:total_row
                sale_row= sales(row,:);
        		prob_row= prob_matrix(row,:);

                value = value + sum(abs((sale_row(sale_row~=0)-prob_row(sale_row~=0)))./sale_row(sale_row~=0))/nnz(sale_row);
            end
            map_value = value/total_row;
        end
    
        function ll_value = logLL(obj, offersets, sales)
            % function to compute the log-likelihood value of the data
            prob_matrix = obj.compute_proba(offersets);
            ll_value = sum(sales(sales~=0).*log(prob_matrix(sales~=0)));
        end

        function ll_val = opt_func(obj, x, offersets, sales)
            % function needed to call fmincon for estimating model
            % parameters
            obj.mean_utils = x(1:obj.num_prods);
            cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
            for level=1:obj.num_levels
                nest_dissim_params{level} = x(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
            end
            obj.nest_dissim_params = nest_dissim_params;
            ll_val = -obj.logLL(offersets, sales);
        end %F1-F2


        function opt_val = fw_opt_fun(obj, alpha, direc, curr_taus, offersets, sales, node_sales, nablaF2)
            new_taus = {};
            for level = 1:obj.num_levels-1
                new_taus{level} = (1 - alpha)*curr_taus{level} + alpha*direc{level};
                obj.nest_dissim_params{level} = exp(new_taus{level});
            end
            v_wts = obj.feedforward(offersets);
            opt_val = 0;
            for level = 2:obj.num_levels+1
                opt_val = opt_val + sum(sum(node_sales{level}.*log(v_wts{level}+(v_wts{level}==0)), 1)./obj.nest_dissim_params{level-1}');
                if(level~=obj.num_levels+1)
                    opt_val = opt_val - sum(new_taus{level-1}.*exp(curr_taus{level-1}).*nablaF2{level-1}');
                end
            end            
        end
    
        function opt_val = pgd_opt_fun_lin(obj, x, before_transform_x, curr_taus, offersets, sales, node_sales, nablaF2)
            new_taus = {};
            obj.mean_utils = x(1:obj.num_prods);
            %obj.mean_utils = curr_mu;
            cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            obj.nest_dissim_params{level} = before_transform_x(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end

             v_wts = obj.feedforward(offersets);
             %F1
            % F1 = 0;
            % for level = 2:obj.num_levels+1
            %     F1 = F1 + sum(sum(node_sales{level}.*log(v_wts{level}+(v_wts{level}==0)), 1)./obj.nest_dissim_params{level-1}');  
            % end
             %F2
             %F2 = F1 -(-obj.logLL(offersets, sales));
            F2=0;
            
            cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
            for level=1:obj.num_levels
                nest_dissim_params{level} = x(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                new_taus{level} = log(nest_dissim_params{level});
            end
            obj.nest_dissim_params = nest_dissim_params;
            
            v_wts = obj.feedforward(offersets);
            
            opt_val = 0;
            for level = 2:obj.num_levels+1
                opt_val = opt_val + sum(sum(node_sales{level}.*log(v_wts{level}+(v_wts{level}==0)), 1)./obj.nest_dissim_params{level-1}');
                if(level~=obj.num_levels+1)
                    opt_val = opt_val - sum(new_taus{level-1}.*exp(curr_taus{level-1}).*nablaF2{level-1}');
                end
            end
             opt_val = opt_val - F2;
        end %F1-linear F2,on delta
        
        function opt_val = pgd_opt_fun_lin_delta(obj, x, curr_taus, offersets, sales, node_sales, nablaF2)
            new_taus = {};
            
            cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
            for level=1:obj.num_levels
                obj.nest_dissim_params{level} = x(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                new_taus{level} = log(obj.nest_dissim_params{level});
            end
            
            v_wts = obj.feedforward(offersets);
            
            opt_val = 0;
            for level = 2:obj.num_levels+1
                opt_val = opt_val + sum(sum(node_sales{level}.*log(v_wts{level}+(v_wts{level}==0)), 1)./obj.nest_dissim_params{level-1}');
                if(level~=obj.num_levels+1)
                    opt_val = opt_val - sum(new_taus{level-1}.*exp(curr_taus{level-1}).*nablaF2{level-1}');
                end
            end
        end %F1-linear F2,on delta only
	
    	function opt_val = pgd_all_opt_fun_lin(obj, x, before_transform_x, curr_taus, current_mu, offersets, sales, node_sales, nablaF2mu, nablaF2)
            new_taus = {};
            obj.mean_utils = before_transform_x(1:obj.num_prods);
            old_theta = log(obj.mean_utils);
            cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            obj.nest_dissim_params{level} = before_transform_x(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end

             v_wts = obj.feedforward(offersets);
             %F1
            % F1 = 0;
            % for level = 2:obj.num_levels+1
            %     F1 = F1 + sum(sum(node_sales{level}.*log(v_wts{level}+(v_wts{level}==0)), 1)./obj.nest_dissim_params{level-1}');  
            % end
             %F2
            % F2 = F1 -(-obj.logLL(offersets, sales));
             F2=0;
            obj.mean_utils = x(1:obj.num_prods);
            new_theta = log(obj.mean_utils);
            cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
            for level=1:obj.num_levels
                nest_dissim_params{level} = x(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                new_taus{level} = log(nest_dissim_params{level});
            end
            obj.nest_dissim_params = nest_dissim_params;            
            v_wts = obj.feedforward(offersets);
            
            opt_val = 0;
            for level = 2:obj.num_levels+1
                opt_val = opt_val + sum(sum(node_sales{level}.*log(v_wts{level}+(v_wts{level}==0)), 1)./obj.nest_dissim_params{level-1}');
                if(level~=obj.num_levels+1)
                    opt_val = opt_val - sum(new_taus{level-1}.*exp(curr_taus{level-1}).*nablaF2{level-1}');
                end
            end
             opt_val = opt_val - F2;
             opt_val = opt_val - sum((new_theta-old_theta).*current_mu.*nablaF2mu);
        end %F1-linear F2, on delta, theta

    	function op = fit(obj, offersets, sales, mu_lb, mu_ub, lambda_lb, method)
            % function to estimate parameters of the model
            % offersets: Q x n 0-1 matrix denoting set membership
            % sales: Q x n matrix with observed sales numbers
            % mu_lb: lower bound for the mean utilities
            % mu_ub: upper bound for the mean utilities
            % lambda_lb: lower bound for the nest dissim params
            % method: 'mm' or 'fmincon'
            
            if(strcmp(method, 'fmincon'))
                fval = obj.fit_fmincon(offersets, sales, mu_lb, mu_ub, lambda_lb);
                op = fval;
            elseif(strcmp(method, 'mm'))
                op = obj.fit_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'knitro'))
                op = obj.fit_knitro(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'knitrobox'))
                op = obj.fit_knitrobox(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fw_mm'))
                op = obj.fit_fw_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdmu'))
                op = obj.fit_gdmu(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdnll1'))
                op = obj.fit_gdnll1(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdnll2'))
                op = obj.fit_gdnll2(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullnll1'))
                op = obj.fit_fullnll1_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullnll2'))
                op = obj.fit_fullnll2_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdlin1'))
                op = obj.fit_gdlin1(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdlin2'))
                op = obj.fit_gdlin2(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fulllin1'))
                op = obj.fit_fulllin1_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fulllin2'))
                op = obj.fit_fulllin2_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fulllin2_3mm'))
                op = obj.fit_fulllin2_3mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'stepwiselin1'))
                op = obj.fit_stepwiselin1_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullback_mm'))
                op = obj.fit_fullback_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullback2_mm'))
                op = obj.fit_fullback2_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullback200_mm'))
                op = obj.fit_fullback200_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullback005_mm'))
                op = obj.fit_fullback005_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullbacknll_mm'))
                op = obj.fit_fullbacknll_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullbacknll200_mm'))
                op = obj.fit_fullbacknll200_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fullbacknll005_mm'))
                op = obj.fit_fullbacknll005_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdnll_mudelta'))
                op = obj.fit_gdnll_mudelta(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdnll200_mudelta'))
                op = obj.fit_gdnll200_mudelta(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdnll005_mudelta'))
                op = obj.fit_gdnll005_mudelta(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdlin_mudelta'))
                op = obj.fit_gdlin_mudelta(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdlin200_mudelta'))
                op = obj.fit_gdlin200_mudelta(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gdlin005_mudelta'))
                op = obj.fit_gdlin005_mudelta(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'full1_mm'))
                op = obj.fit_full1_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
 	        elseif(strcmp(method, 'agd_mu_delta'))
                op = obj.fit_agd_mu_delta(offersets, sales, mu_lb, mu_ub, lambda_lb);            
            elseif(strcmp(method, 'test'))
                op = obj.fit_test(offersets, sales, mu_lb, mu_ub, lambda_lb);

            end            
        end        
                  
        function probs = local_probs(obj, level, v_wts)
            % helper function to compute local probability at a level
            lambda_parent = obj.adj_mats{level}*obj.nest_dissim_params{level};
            
            %normalize vs before raising them to an exponent
            vs = v_wts{level};
            normalize_factor = vs*obj.adj_mats{level};
            repl_normalize_factor = normalize_factor*obj.adj_mats{level}';
            normalized_vs = vs./(repl_normalize_factor  + (repl_normalize_factor  == 0)); % ensures that the exponents don't blow up
            
            wts = bsxfun(@power, normalized_vs, 1./lambda_parent');
            
            total_wts = wts*obj.adj_mats{level};
            repl_total_wts = total_wts*obj.adj_mats{level}'; % replicate wts so that it is easy to divide
            probs = wts./(repl_total_wts + (repl_total_wts==0)); % Q x V_l
        end
        
        function [gammaiq, betaiq] = update_gamma_beta(obj, node_sales, v_wts)
            num_offersets = size(v_wts{1}, 1);
            % update gammaiq and betaiq
%             gammaiq = {node_sales{obj.num_levels+1}};
            gammaiq = {zeros(num_offersets, 1)};
            tilde_betaiq = {zeros(num_offersets, 1)}; % tilde_beta = beta/lambda_parent
            betaiq = {zeros(num_offersets, 1)};
            for level=obj.num_levels:-1:1
                rev_level = obj.num_levels-level+2; % reverse level index
                % replicate the parent parameters to the shape of children
                % nodes to facilitate matrix operations
                repl_lambda_parent = obj.adj_mats{level}*obj.nest_dissim_params{level}; % V_l x 1
                repl_gamma_parent = gammaiq{rev_level-1}*obj.adj_mats{level}'; %Q x V_l
                % compute one-level probabilities
                probs = obj.local_probs(level, v_wts); % Q x V_l
                % repl_total_wts: Q x V_l matrix
                % gammaiq{level+1} = ga_first_term + ga_second_term

                ga_first_term = bsxfun(@rdivide, node_sales{level}, repl_lambda_parent');
                ga_second_term = probs.*repl_gamma_parent;
                gammaiq{rev_level} = ga_first_term + ga_second_term;

                be_tmp_term = bsxfun(@rdivide, node_sales{level+1}, obj.nest_dissim_params{level}') + tilde_betaiq{rev_level-1};
                repl_be_tmp_term = be_tmp_term*obj.adj_mats{level}';
                tilde_betaiq{rev_level} = probs.*repl_be_tmp_term;
                betaiq{rev_level} = bsxfun(@times, tilde_betaiq{rev_level}, repl_lambda_parent');
            end
            gammaiq = fliplr(gammaiq);
            tilde_betaiq = fliplr(tilde_betaiq);
            betaiq = fliplr(betaiq);
        end
            
        function [gamma1iq, gammaiq, betaiq] = update_gamma1_gamma_beta(obj, node_sales, v_wts)
            num_offersets = size(v_wts{1}, 1);
            % update gamma1iq, gammaiq, and betaiq
%             gammaiq = {node_sales{obj.num_levels+1}};
            gamma1iq = {bsxfun(@rdivide, node_sales{obj.num_levels+1}, obj.nest_dissim_params{obj.num_levels})};
            gammaiq = {zeros(num_offersets, 1)};
            tilde_betaiq = {zeros(num_offersets, 1)}; % tilde_beta = beta/lambda_parent
            betaiq = {zeros(num_offersets, 1)};
            for level=obj.num_levels:-1:1
                rev_level = obj.num_levels-level+2; % reverse level index
                % replicate the parent parameters to the shape of children
                % nodes to facilitate matrix operations
                repl_lambda_parent = obj.adj_mats{level}*obj.nest_dissim_params{level}; % V_l x 1
                repl_gamma_parent = gammaiq{rev_level-1}*obj.adj_mats{level}'; %Q x V_l
                repl_gamma1_parent = gamma1iq{rev_level-1}*obj.adj_mats{level}'; %Q x V_l
                
                % compute one-level probabilities
                probs = obj.local_probs(level, v_wts); % Q x V_l
                % repl_total_wts: Q x V_l matrix
                % gammaiq{level+1} = ga_first_term + ga_second_term
                
                if(level~=1)
                    ga1_first_term = bsxfun(@rdivide, node_sales{level}, obj.nest_dissim_params{level-1}');
                    ga1_second_term = probs.*repl_gamma1_parent;
                    gamma1iq{rev_level} = ga1_first_term + ga1_second_term;
                end

                ga_first_term = bsxfun(@rdivide, node_sales{level}, repl_lambda_parent');
                ga_second_term = probs.*repl_gamma_parent;
                gammaiq{rev_level} = ga_first_term + ga_second_term;

                be_tmp_term = bsxfun(@rdivide, node_sales{level+1}, obj.nest_dissim_params{level}') + tilde_betaiq{rev_level-1};
                repl_be_tmp_term = be_tmp_term*obj.adj_mats{level}';
                tilde_betaiq{rev_level} = probs.*repl_be_tmp_term;
                betaiq{rev_level} = bsxfun(@times, tilde_betaiq{rev_level}, repl_lambda_parent');
            end
            gamma1iq = fliplr(gamma1iq);
            gammaiq = fliplr(gammaiq);
            tilde_betaiq = fliplr(tilde_betaiq);
            betaiq = fliplr(betaiq);
        end       
        
        function [fun_val, grad_val] = lambda_opt_func(obj, x, first_coef, second_coef, third_coef, children, parent_lambda)
            % function to evalute the objective value for opt problem
            % corresponding to each lambda
            y = exp(x);
            first_term = sum(first_coef.*(sum(children.^(1/y), 2).^(y/parent_lambda)));
            tmp_second_term = second_coef.*(children.^(1/y));
            second_term = sum(tmp_second_term(:));
            third_term = x*third_coef;
            
            fun_val = first_term + second_term - third_term;
            
            if nargout > 1 % compute the gradient
                grad_first_fn  = first_coef.*(sum(children.^(1/y), 2).^(y/parent_lambda));
                log_term = sum(children.^(1/y), 2).^(y/parent_lambda);
                grad_first_first_term = grad_first_fn.*log(log_term + (log_term==0));

                children_sum = sum(children.^(1/y), 2);
                children_sum_log = sum((children.^(1/y)).*log(children + (children==0)), 2);

                grad_first_second_term = (1/parent_lambda)*(grad_first_fn(children_sum~=0)).*(children_sum_log(children_sum~=0)./children_sum(children_sum~=0));

                grad_first_term = sum(grad_first_first_term(:)) - sum(grad_first_second_term(:));

                grad_tmp_second_term = tmp_second_term.*log(children + (children==0));
                grad_second_term = -(1/y)*sum(grad_tmp_second_term(:));
                grad_third_term = third_coef;

                grad_val = grad_first_term + grad_second_term - grad_third_term;
%                 fprintf('%f, %f, %f\n', x, fun_val, grad_val);
            end
            
        end
        
        function new_lambdas = opt_lambda(obj, level, v_wts, gammaiq, betaiq, node_sales)
            parent_lambdas = obj.adj_mats{level}*obj.nest_dissim_params{level};
            
            % compute the coefficients for the first term
            tmp_exps = bsxfun(@power, v_wts{level}, 1./parent_lambdas');
            first_coef = betaiq{level}./(tmp_exps + (tmp_exps==0));
            
            % compute the coefficients for the second term
            probs = obj.local_probs(level-1, v_wts);
            curr_lambdas = obj.adj_mats{level-1}*obj.nest_dissim_params{level-1};
            tmp_exps = bsxfun(@power, v_wts{level-1}, 1./curr_lambdas');
            repl_sales = node_sales{level}*obj.adj_mats{level-1}';
            
            second_coef = repl_sales.*(probs./(tmp_exps + (tmp_exps==0)));
            
            % compute coefficients for the third term
%             tmp_exp_sums = tmp_exps*obj.adj_mats{level-1};
%             Delta_first_term = log(tmp_exp_sums + (tmp_exp_sums==0));
            Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
            tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
            Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
            Delta = Delta_first_term - Delta_second_term;
            % derivative of F2
            nablaF2_first_term = gammaiq{level}.*Delta;
            tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
            nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
            nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
            
            third_coef = nablaF2.*obj.nest_dissim_params{level-1}';
            
            new_lambdas = zeros(1, obj.num_nodes_per_level(level));
            if(level == 2)
                child_lambdas = exp(-5)*ones(obj.num_nodes_per_level(level), 1);
            else
                child_lambdas = max(bsxfun(@times, obj.adj_mats{level-1}, obj.nest_dissim_params{level-2}))';
            end

            for i=1:obj.num_nodes_per_level(level)
                child_indices = (obj.adj_mats{level-1}(:,i) == 1);
                f = @(x) obj.lambda_opt_func(x, first_coef(:,i), second_coef(:,child_indices), third_coef(i), v_wts{level-1}(:,child_indices), parent_lambdas(i));
                options_con = optimoptions('fmincon', 'Display', 'off', 'MaxFunEvals',50000);
                options = optimoptions('fminunc', 'Display', 'off', 'MaxFunEvals',50000, 'Algorithm', 'quasi-newton');
                options_unc = optimoptions('fminunc', 'Display', 'off', 'MaxFunEvals',50000, 'Algorithm', 'trust-region', 'GradObj','on');
                
%                 if(level==2)
%                     lb = -Inf;
%                 else
%                     lb = log(max(obj.nest_dissim_params{level-2}(child_indices)));
%                 end
%                 ub = log(parent_lambdas(i));
                x0 = log(curr_lambdas(i));
%                 [x, fval, exitflag, output] = fmincon(f, x0, [], [], [], [], -Inf, Inf, [], options_con);
%                 [x, fval, exitflag, output] = fminunc(f, x0, options_unc);
%                 x = fminbnd(f, log(child_lambdas(i)), log(parent_lambdas(i)));
                x = fminbnd(f, -5, log(parent_lambdas(i)));                
%                 if(f(x) > f(x0))
%                     x = x0;
%                 end 
                assert(x <= log(parent_lambdas(i)));
%                 assert(f(x) <= f(x0));
%                 [x, fval, exitflag, output] = fminunc(f, x0, options);
%                 x = fminsearch(f, x0);
                new_lambdas(i) = exp(x);
            end
        end
        
        function curr_obj = fit_fw_mm(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            % uses MM algo to estimate the parameters of the model
            
            % compute sales at each node in the tree and initialize the
            % parameters of the model
            obj.mean_utils = ones(1, obj.num_prods);
            node_sales = {sales};
            for level=1:obj.num_levels
                node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
            end
            obj.nest_dissim_params{obj.num_levels} = 1;
            
            prev_obj = -obj.logLL(offersets, sales);
            N_ITERS = obj.fwmm_niters;
            num_offersets = size(offersets, 1);
            v_wts = obj.feedforward(offersets);
            [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
            
            for n_iter=1:N_ITERS                
                % update the values of mean utilities
                nablaF2mu = sum(gammaiq{1}, 1);
                repl_lambda_parent = obj.adj_mats{1}*obj.nest_dissim_params{1};
                mu_terms = (obj.mean_utils.*exp(obj.mean_utils./repl_lambda_parent')).*nablaF2mu;
                beta_sums = sum(betaiq{1}, 1);
                tmp_mean_utils = repl_lambda_parent'.*Lambert_W(mu_terms./beta_sums);
                obj.mean_utils = tmp_mean_utils;
%                 obj.mean_utils = max(tmp_mean_utils, 1); % make sure that the means are > 1
%                 obj.mean_utils = 1 + obj.mean_utils - min(obj.mean_utils);
                % update vs, betas, and gammas
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                curr_obj = -obj.logLL(offersets, sales);
%                 assert(round(prev_obj - curr_obj, 7) >= 0);
                if(~(round(prev_obj - curr_obj, 7) >= 0))
                    fprintf('%d,%f\n',n_iter,prev_obj-curr_obj);
                    break;
                end

                %update values of lambdas through one iteration of FW
                Z_star = {};
                nablaF2_list = {};
                curr_taus = {};
                for level=2:obj.num_levels
                    curr_taus{level-1} = log(obj.nest_dissim_params{level-1});
                    probs = obj.local_probs(level-1, v_wts);
                    % compute Delta
                    Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
                    tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                    Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
                    Delta = Delta_first_term - Delta_second_term;
                    % derivative of F2
                    nablaF2_first_term = gammaiq{level}.*Delta;
                    tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                    nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
                    nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
                    nablaF2_list{level-1} = nablaF2;
                    % derivative of F1
                    nablaF1_first_term = gamma1iq{level-1}.*Delta;
                    nablaF1_second_term = bsxfun(@rdivide, node_sales{level}.*log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}'.^2);
                    nablaF1 = sum(nablaF1_first_term - nablaF1_second_term, 1);
                    
                    fw_c = -obj.nest_dissim_params{level-1}'.*(nablaF1 - nablaF2);
                    if(level==2)
                        Z_star{level-1} = fw_c;
                    else
                        Z_star{level-1} = fw_c + max(0, Z_star{level-2})*obj.adj_mats{level-1};
                    end            
                end
                % determine the FW direction
                x_direc = {[1]};
                direc = {};
                for level=2:obj.num_levels
                    rev_level = obj.num_levels-level+2;
                    x_direc{level} = ((Z_star{rev_level-1} >= 0).*(obj.adj_mats{rev_level}*(x_direc{level-1} == 1))')';
                    direc{level-1} = (x_direc{level}-1)*log(1/lambda_lb);
                end
                direc = fliplr(direc);
                
                f = @(x) obj.fw_opt_fun(x, direc, curr_taus, offersets, sales, node_sales, nablaF2_list);
                alpha = fminbnd(f, 0, 1);
                for level = 1:obj.num_levels-1
                    new_taus = (1 - alpha)*curr_taus{level} + alpha*direc{level};
                    obj.nest_dissim_params{level} = exp(new_taus);
                end
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                

                curr_obj = -obj.logLL(offersets, sales);
%                 fprintf('%d,%f,%f\n',n_iter, prev_obj, curr_obj);
                if(~(round(prev_obj - curr_obj, 7) >= 0))
                    fprintf('%d,%f\n',n_iter,prev_obj-curr_obj);
                    break;
                end
                
                % check stopping condition
                if(prev_obj - curr_obj < 1e-16)
                    break;
                end
                
                prev_obj = curr_obj;
%                 n_iter, prev_obj, curr_obj
            end
            fprintf('Algo. termininated in %d iters...obj val %f\n', n_iter, curr_obj);
        end
        
        function fval = fit_knitro(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            f = @(x) obj.opt_func(x, offersets, sales);
            
            % x variable: [mean_utils, dissim_param_level_1, ...,
            % dissim_param_level_d]
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            lb = [mu_lb*ones(1,obj.num_prods), lambda_lb*ones(1,num_non_leaf_nodes-1), 1];
            ub = [mu_ub*ones(1,obj.num_prods), ones(1,num_non_leaf_nodes)];
            lb(1) = 0;
            ub(1) = 0;
            %options = optimoptions('fmincon', 'Display', 'off');
            %options = optimoptions('fmincon','MaxFunEvals',50000);
%             options = optimoptions('fmincon', 'Display', 'off', 'MaxFunEvals',50000);
            % construct the inequality constraints for the
            % nest_dissim_params
            Aleq = [];
            for level=1:obj.num_levels-1
                local_A = [eye(obj.num_nodes_per_level(level+1)), -obj.adj_mats{level+1}];
                [row1, col1] = size(Aleq);
                [row2, col2] = size(local_A);
                num_new_vars = obj.num_nodes_per_level(level+2);
                if(isempty(Aleq))
                    Aleq = local_A;
                else
                    Aleq = [Aleq, zeros(row1, num_new_vars); zeros(row2, col1+num_new_vars-col2), local_A];
                end
            end
            % add the columns corresponding to the leaf nodes
            Aleq = [zeros(size(Aleq, 1), obj.num_prods), Aleq];
            bleq = zeros(size(Aleq, 1), 1);
            x0 = [zeros(1, obj.num_prods), ones(1, num_non_leaf_nodes)];
            options = knitro_options('maxit', 8, 'maxtime_real', 1e4, 'xtol', 1e-5);
            [x, fval, exitflag, output, lambda, grad, hessian] = knitro_nlp(f, x0, Aleq, bleq, [], [], lb, ub, [], [],options);            
            

			obj.mean_utils = x(1:obj.num_prods);
                        cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            nest_dissim_params{level} = x(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end
                        obj.nest_dissim_params = nest_dissim_params;           
        end
        

	function fval = fit_test(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
		num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
		obj.mean_utils = ones(1, obj.num_prods);
                node_sales = {sales};
                for level=1:obj.num_levels
                    node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                    obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
                end
                obj.nest_dissim_params{obj.num_levels} = 1;


            fval =10000;
                       
        end


        function fval = fit_knitrobox(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            
            Box = -eye(num_non_leaf_nodes);
            Box_i = 0;
            for level=2:obj.num_levels
                for node=1:obj.num_nodes_per_level(level)
                    Box_i = Box_i+1;
                    Box_pr = obj.adj_mats{level};
                    Box_j = sum(obj.num_nodes_per_level(2:level))+find(Box_pr(node,:));
                    Box(Box_i,Box_j) = 1;
                end
            end
            
            Box_tran = [eye(obj.num_prods),zeros(obj.num_prods,num_non_leaf_nodes);zeros(num_non_leaf_nodes,obj.num_prods),Box]';
            
%             for level=1:obj.num_levels-1
%                 curr_lambdas = obj.nest_dissim_params{level};
%                 parent_lambdas = obj.adj_mats{level+1}*obj.nest_dissim_params{level+1};
%                 if(all(curr_lambdas <= parent_lambdas) == 0)
%                    fval =100000;
%                    fprintf('Solution obtained by fmincon infeasible...\n');                    
%                 end
%                 assert(all(curr_lambdas <= parent_lambdas));
%             end
            
            f = @(x) obj.opt_func(exp(x/Box_tran), offersets, sales);
            
            % x variable: [mean_utils, dissim_param_level_1, ...,
            % dissim_param_level_d]
            
            lb = [zeros(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];
            ub = [log(2+exp(1))*ones(1,obj.num_prods), (-log(lambda_lb))*ones(1,num_non_leaf_nodes-1),0];
            lb(1) = 1;
            ub(1) = 1;
            
%             Aleq = [];
%             for level=1:obj.num_levels-1
%                 local_A = [eye(obj.num_nodes_per_level(level+1)), -obj.adj_mats{level+1}];
%                 [row1, col1] = size(Aleq);
%                 [row2, col2] = size(local_A);
%                 num_new_vars = obj.num_nodes_per_level(level+2);
%                 if(isempty(Aleq))
%                     Aleq = local_A;
%                 else
%                     Aleq = [Aleq, zeros(row1, num_new_vars); zeros(row2, col1+num_new_vars-col2), local_A];
%                 end
%             end
            % add the columns corresponding to the leaf nodes
%             Aleq = [zeros(size(Aleq, 1), obj.num_prods), Aleq];
%             bleq = zeros(size(Aleq, 1), 1);
            x0 = [ones(1, obj.num_prods), zeros(1,num_non_leaf_nodes)];
            
            options = knitro_options('maxit', 2000, 'maxtime_cpu', 6e2, 'xtol', 1e-5);
            %x0 = [zeros(1, obj.num_prods+num_non_leaf_nodes-1), 1];
            [x, fval, exitflag, output, lambda, grad, hessian] = knitro_nlp(f, x0, [], [], [], [], lb, ub, [], [], options); 
            
            % make sure that the lambda constraints are satisfied
            
            
            if(all(x>=lb)==0)
                %fval = 100000;
                fprintf('infeasible...\n'); 
            end
            %[x, fval, exitflag, output] = fmincon(f, zeros(1,obj.num_prods+obj.num_nests), [], [], [], [], lb, ub, []);            
        end
                     
        function fval = fit_knitrotran(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            f = @(x) obj.opt_func(exp(x), offersets, sales);
            
            % x variable: [mean_utils, dissim_param_level_1, ...,
            % dissim_param_level_d]
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            lb = [zeros(1,obj.num_prods), log(lambda_lb)*ones(1,num_non_leaf_nodes-1), 0];
            ub = [(log(2+exp(1)))*ones(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];
            lb(1) = 1;
            ub(1) = 1;
            
            Aleq = [];
            for level=1:obj.num_levels-1
                local_A = [eye(obj.num_nodes_per_level(level+1)), -obj.adj_mats{level+1}];
                [row1, col1] = size(Aleq);
                [row2, col2] = size(local_A);
                num_new_vars = obj.num_nodes_per_level(level+2);
                if(isempty(Aleq))
                    Aleq = local_A;
                else
                    Aleq = [Aleq, zeros(row1, num_new_vars); zeros(row2, col1+num_new_vars-col2), local_A];
                end
            end
            % add the columns corresponding to the leaf nodes
            Aleq = [zeros(size(Aleq, 1), obj.num_prods), Aleq];
            bleq = zeros(size(Aleq, 1), 1);
            x0 = [ones(1, obj.num_prods), zeros(1,num_non_leaf_nodes)];
            options = knitro_options('maxit', 2000, 'maxtime_real', 6e2, 'xtol', 1e-5);
            %x0 = [zeros(1, obj.num_prods+num_non_leaf_nodes-1), 1];
            [x, fval, exitflag, output, lambda, grad, hessian] = knitro_nlp(f, x0, Aleq, bleq, [], [], lb, ub, [], [], options); 
            
            % make sure that the lambda constraints are satisfied
            for level=1:obj.num_levels-1
                curr_lambdas = obj.nest_dissim_params{level};
                parent_lambdas = obj.adj_mats{level+1}*obj.nest_dissim_params{level+1};
                if(all(curr_lambdas <= parent_lambdas) == 0)
                   %fval =100000;
                   fprintf('Solution obtained by fmincon infeasible...\n');                    
                end
%                 assert(all(curr_lambdas <= parent_lambdas));
            end
            
            %[x, fval, exitflag, output] = fmincon(f, zeros(1,obj.num_prods+obj.num_nests), [], [], [], [], lb, ub, []);            
        end 
                       
        function fval = fit_gdmu(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            % uses gd algo to estimate the parameters of the model
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
           
            % Define the objective function 
            f = @(x) obj.opt_func(x, offersets, sales);
            
            % x variable: [mu, lambda...]
            
            % Constraints           
            lb = [mu_lb*ones(1,obj.num_prods), zeros(1,num_non_leaf_nodes-1), 1];

            ub = [(Inf)*ones(1,obj.num_prods), ones(1,num_non_leaf_nodes)];
            
            Aleq = [];
            for level=1:obj.num_levels-1
                local_A = [eye(obj.num_nodes_per_level(level+1)), -obj.adj_mats{level+1}];
                [row1, col1] = size(Aleq);
                [row2, col2] = size(local_A);
                num_new_vars = obj.num_nodes_per_level(level+2);
                if(isempty(Aleq))
                    Aleq = local_A;
                else
                    Aleq = [Aleq, zeros(row1, num_new_vars); zeros(row2, col1+num_new_vars-col2), local_A];
                end
            end
            % add the columns corresponding to the leaf nodes
            Aleq = [zeros(size(Aleq, 1), obj.num_prods), Aleq];
            bleq = zeros(size(Aleq, 1), 1);
            x0 = [ones(1, obj.num_prods), ones(1, num_non_leaf_nodes)];
            
            % GD interation options
            %tol = 1e-16;
            maxiter = 49;
            
            % initialize 
            x = x0;
            
            niter = 0;
            %dfval = Inf;
                       
            % compute sales at each node in the tree and initialize the
                % parameters of the model
                obj.mean_utils = ones(1, obj.num_prods);
                node_sales = {sales};
                for level=1:obj.num_levels
                    node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                    obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
                end
                obj.nest_dissim_params{obj.num_levels} = 1;
                        
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);

            curr_obj = -obj.logLL(offersets, sales);
            curr_time = toc;
            fprintf(2,'PGD neglog at iter 0 is %f with time %f\n', curr_obj, curr_time);
            % Gradient descent
            while (niter <= maxiter)
                nablaF2mu = sum(gammaiq{1}, 1);
                repl_lambda_parent = obj.adj_mats{1}*obj.nest_dissim_params{1};

                beta_sums = sum(betaiq{1}, 1);
                current_mu = obj.mean_utils;
                gradient_mu = (beta_sums./repl_lambda_parent'-nablaF2mu);
                                       
                    %update values of lambdas through one iteration of FW
                    Z_star = {};
                    nablaF2_list = {};
                    nablaNLL = {};
                    curr_taus = {};
                    for level=2:obj.num_levels
                        curr_taus{level-1} = log(obj.nest_dissim_params{level-1});
                        probs = obj.local_probs(level-1, v_wts);
                        % compute Delta
                        Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
                        tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
                        Delta = Delta_first_term - Delta_second_term;
                        % derivative of F2
                        nablaF2_first_term = gammaiq{level}.*Delta;
                        tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
                        nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
                        nablaF2_list{level-1} = nablaF2;
                        % derivative of F1
                        nablaF1_first_term = gamma1iq{level-1}.*Delta;
                        nablaF1_second_term = bsxfun(@rdivide, node_sales{level}.*log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}'.^2);
                        nablaF1 = sum(nablaF1_first_term - nablaF1_second_term, 1);

                        %fw_c = -obj.nest_dissim_params{level-1}'.*(nablaF1 - nablaF2);
                        nablaNLL{level-1} = nablaF1 - nablaF2;
                        
%                         if(level==2)
%                             Z_star{level-1} = fw_c;
%                         else
%                             Z_star{level-1} = fw_c + (Z_star{level-2})*obj.adj_mats{level-1};
%                         end            
                    end
                    
                    gradient_lambda = reshape(cell2mat(nablaNLL)',[],1);
                    current_gradient = [gradient_mu,gradient_lambda',0];
                    
                    current_lambda = reshape(cell2mat(obj.nest_dissim_params'),[],1);
                    % mu, lambda
                    x = [current_mu,current_lambda']; 
 
                    options = knitro_options('maxit', 800, 'maxtime_real', 5e2, 'xtol', 1e-5);
                                                    
                    % Initial step size
                    alpha = 0.5;

                    % take step:
                    xnew = x - alpha*current_gradient;

                    %projection
                    f_projection = @(y) norm(y-xnew')^2;
                    y0 = [zeros(1,obj.num_prods), zeros(1,num_non_leaf_nodes-1), 1]';
                    [xpro, fval, exitflag, output, lambda, grad, hessian] = knitro_nlp(f_projection, y0, Aleq, bleq, [], [], lb, ub, [],[], options);
                    
                     while(any(xpro(obj.num_prods+1:length(x))<1e-3))                            
                                alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            %projection
                            f_projection = @(y) norm(y-xnew')^2;
                            y0 = [zeros(1,obj.num_prods), zeros(1,num_non_leaf_nodes-1), 1]';
                            [xpro, fval, exitflag, output, lambda, grad, hessian] = knitro_nlp(f_projection, y0, Aleq, bleq, [], [], lb, ub, [],[], options);
                     end
                    
                    if (all(xpro))
                    Stopping_Condition = f(xpro')>f(x)+current_gradient*((xpro'-x)')+1/(2*alpha)*norm(xpro'-x)^2;
                    else
                    Stopping_Condition = 1;
                    end
                    
                     
                     while (Stopping_Condition)
                         % x = xnew;
                         alpha = alpha/2;
                         xnew = x - alpha*current_gradient;
                         f_projection = @(y) norm(y-xnew')^2;
                         y0 = [ones(1, obj.num_prods), ones(1, num_non_leaf_nodes)]';
                         [xpro, fval, exitflag, output, lambda, grad, hessian] = knitro_nlp(f_projection, y0, Aleq, bleq, [], [], lb, ub, [],[], options);
                         
                         if (all(xpro))
                            Stopping_Condition = f(xpro')>f(x)+current_gradient*((xpro'-x)')+1/(2*alpha)*norm(xpro'-x)^2;
                         else
                            Stopping_Condition = 1;
                         end

                     end
                    
                    
                    % check step
                    if ~isfinite(xpro)
                        display(['Number of iterations: ' num2str(niter)])
                        error('x is inf or NaN')
                    end
                    %fprintf('niter %f\n', niter);
                    % update termination
                    % metrics
                    niter = niter + 1;
                    %dfval = norm(f(x)-f(xpro'));
                    
                        update_xnew = xpro';
                        obj.mean_utils = update_xnew(1:obj.num_prods);
                        cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            nest_dissim_params{level} = update_xnew(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end
                        obj.nest_dissim_params = nest_dissim_params;
                        v_wts = obj.feedforward(offersets);
                        [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                        
                        curr_obj = -obj.logLL(offersets, sales);
                        %fprintf('niter %f, value %f\n', niter,curr_obj);
                        curr_time = toc;
                        fprintf(2,'PGD neglog at iter %d is %f with time %f\n', niter, curr_obj, curr_time);

                    %x = update_xnew;
            end

            %display(['Number of iterations: ' num2str(niter)])
            fval = curr_obj;
            fprintf('GD mu NLL Algo. termininated in %d iters...obj val %f\n', niter, fval);
        end %gd back tracking 1d NLL mu lambda 

	function fval = fit_null(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            		fval = 20000;
               	     end %NULL



        
        function fval = fit_fullback_mm(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            % Box transformation matrix
            Box = -eye(num_non_leaf_nodes);
            Box_i = 0;
            for level=2:obj.num_levels
                for node=1:obj.num_nodes_per_level(level)
                    Box_i = Box_i+1;
                    Box_pr = obj.adj_mats{level};
                    Box_j = sum(obj.num_nodes_per_level(2:level))+find(Box_pr(node,:));
                    Box(Box_i,Box_j) = 1;
                end
            end            
            Box_tran = [eye(obj.num_prods),zeros(obj.num_prods,num_non_leaf_nodes);zeros(num_non_leaf_nodes,obj.num_prods),Box]';
          
            % x variable: [mu, delata...]
            
            % Box constraints on mu, delta
            lb = [(-Inf)*ones(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];          
            
            % GD interation options
            %tol = 1e-20;
            maxiter = 399;
            
            niter = 0;
            %dfval = Inf;


            % compute sales at each node in the tree and initialize the
                % parameters of the model
                obj.mean_utils = ones(1, obj.num_prods);
                node_sales = {sales};
                for level=1:obj.num_levels
                    node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                    obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
                end
                obj.nest_dissim_params{obj.num_levels} = 1;
                        
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);


            curr_obj = -obj.logLL(offersets, sales);
            curr_time = toc;
            fprintf(2,'MM neglog at iter 0 is %f with time %f\n', curr_obj, curr_time);

            % Gradient descent
            b_l = sum(sales,1);
            while (niter <= maxiter)
                repl_lambda_parent = obj.adj_mats{1}*obj.nest_dissim_params{1};                    
                for nbiterations =1:8
                % update the values of mean utilities
                nablaF2mu = sum(gammaiq{1}, 1);                
                beta_sums = sum(betaiq{1}, 1);                                
                a_l = repl_lambda_parent'.* (beta_sums./repl_lambda_parent'-nablaF2mu) + b_l;

                obj.mean_utils = obj.mean_utils + repl_lambda_parent'.*(log(b_l./a_l));
                %obj.mean_utils = obj.mean_utils - obj.mean_utils(1)
                % update vs, betas, and gammas
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                end
                % shifting
                %current_mu = obj.mean_utils;
                %obj.mean_utils = obj.mean_utils - min(obj.mean_utils);
                %v_wts = obj.feedforward(offersets);
                %[gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                %nablaF2mu = sum(gammaiq{1}, 1);
                %beta_sums = sum(betaiq{1}, 1);
                
                %gradient_mu = (beta_sums./repl_lambda_parent'-nablaF2mu);

                %gradient_theta = gradient_mu.*obj.mean_utils;
                    
                    %update values of lambdas through one iteration of GD
                    Z_star = {};
                    nablaF2_list = {};
                    curr_taus = {};
                    for level=2:obj.num_levels
                        curr_taus{level-1} = log(obj.nest_dissim_params{level-1});
                        probs = obj.local_probs(level-1, v_wts);
                        % compute Delta
                        Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
                        tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
                        Delta = Delta_first_term - Delta_second_term;
                        % derivative of F2
                        nablaF2_first_term = gammaiq{level}.*Delta;
                        tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
                        nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
                        nablaF2_list{level-1} = nablaF2;
                        % derivative of F1
                        nablaF1_first_term = gamma1iq{level-1}.*Delta;
                        nablaF1_second_term = bsxfun(@rdivide, node_sales{level}.*log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}'.^2);
                        nablaF1 = sum(nablaF1_first_term - nablaF1_second_term, 1);

                        fw_c = -obj.nest_dissim_params{level-1}'.*(nablaF1 - nablaF2);
                        
                        if(level==2)
                            Z_star{level-1} = fw_c;
                        else
                            Z_star{level-1} = fw_c + (Z_star{level-2})*obj.adj_mats{level-1};
                        end            
                    end
                    
                    gradient_delta = reshape(cell2mat(Z_star)',[],1);

                    % Gradient:
                    %current_gradient = [gradient_theta,gradient_delta',0];
                    current_gradient = [zeros(1,obj.num_prods),gradient_delta',0];
                    %tmp_mu = obj.mean_utils+1-min(obj.mean_utils);
             
                    current_lambda = reshape(cell2mat(obj.nest_dissim_params'),[],1);
                    % mu, lambda
                    %before_transform_x = [tmp_mu,current_lambda']; 
                    % theta, delta
                    %x = log(before_transform_x)*Box_tran;
                    x = [obj.mean_utils,log(current_lambda')*Box'];

                    % Backtracking Step size
                     alpha = 0.5;
                         
                     % take step:
                     xnew = x - alpha*current_gradient;
                     
                     % projection
                     xnew = max([xnew;lb]);
                    
                    
                    f = @(x) obj.pgd_opt_fun_lin_delta([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], curr_taus, offersets, sales, node_sales, nablaF2_list);
                    %f = @(x) obj.pgd_all_opt_fun_lin([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], before_transform_x, curr_taus, current_mu, offersets, sales, node_sales, nablaF2mu, nablaF2_list);
                    
                    %Stopping_Condition = f_transformed(xnew)>f_transformed(x)-alpha/2*norm(current_gradient)^2;
                     fval_old = f(x);
                     
                     
                     while(any(exp(xnew(obj.num_prods+1:length(x))/Box')<1e-3))                            
                                alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            %fprintf('alpha %f\n', alpha*10000);
                     end
                     
                     ready = false;
                    while ~ready
                        try
                            Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                            ready = true;
                        catch ME
                            alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            fprintf('%s\ntrying it again...\n', ME.message);
                        end
                    end
                     
                     while (Stopping_Condition) 
                         % x = xnew;
                         alpha = alpha/2;
                         xnew = x - alpha*current_gradient;
                         xnew = max([xnew;lb]);
                        
                         ready = false;
                        while ~ready
                            try
                                Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                                ready = true;
                            catch ME
                                alpha = alpha*0.5;
                                xnew = x - alpha*current_gradient;
                                xnew = max([xnew;lb]);
                                fprintf('%s\ntrying it again...\n', ME.message);
                            end
                        end
                         %Stopping_Condition = f(xnew)>f(x)+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     end                                      
                     

                         %fprintf('alpha %f, %f, %f\n', alpha(1),alpha(2),f_transformed(xnew));
                     
                    % check step
                    if ~isfinite(xnew)
                        display(['Number of iterations: ' num2str(niter)])
                        error('x is inf or NaN')
                    end
                    %fprintf('niter %f\n', niter);

                    niter = niter + 1;
                    
                        
                         update_xnew = [xnew(1:obj.num_prods),exp(xnew(obj.num_prods+1:length(x))/Box')];
                        
                        cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            obj.nest_dissim_params{level} = update_xnew(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end
                        
                        v_wts = obj.feedforward(offersets);
                        [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                    curr_obj = -obj.logLL(offersets, sales);
                    %fprintf('Found step size %f,and the value of original objective %f\n', alpha,curr_obj);
                    %fprintf('%f, %f\n', alpha,curr_obj);
                    curr_time = toc;
                    fprintf(2,'MM neglog at iter %d is %f with time %f\n', niter, curr_obj, curr_time);

            end
            %display(['Number of iterations: ' num2str(niter)])
            fval = curr_obj;
            fprintf('Fullback2 Lin 1d Algo. termininated in %d iters...obj val %f\n', niter, fval);
        end % mm on mu, pgd on delta backtracking 1d, F1-linearF2 

        function fval = fit_fullbacknll_mm(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            % Box transformation matrix
            Box = -eye(num_non_leaf_nodes);
            Box_i = 0;
            for level=2:obj.num_levels
                for node=1:obj.num_nodes_per_level(level)
                    Box_i = Box_i+1;
                    Box_pr = obj.adj_mats{level};
                    Box_j = sum(obj.num_nodes_per_level(2:level))+find(Box_pr(node,:));
                    Box(Box_i,Box_j) = 1;
                end
            end            
            Box_tran = [eye(obj.num_prods),zeros(obj.num_prods,num_non_leaf_nodes);zeros(num_non_leaf_nodes,obj.num_prods),Box]';
          
            % x variable: [theta, delata...]
            
            % Box constraints on mu, delta
            lb = [(-Inf)*ones(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];
            
            % Initial value
            x0 = [ones(1, obj.num_prods), zeros(1,num_non_leaf_nodes)];
            
            % GD interation options
            %tol = 1e-20;
            maxiter = 399;
            
            % initialize 
            x = x0;
            
            niter = 0;
            %dfval = Inf;
           
            
            % compute sales at each node in the tree and initialize the
                % parameters of the model
                obj.mean_utils = ones(1, obj.num_prods);
                node_sales = {sales};
                for level=1:obj.num_levels
                    node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                    obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
                end
                obj.nest_dissim_params{obj.num_levels} = 1;
                        
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                %prev_obj = -obj.logLL(offersets, sales);
            % Gradient descent
            b_l = sum(sales,1);
            while (niter <= maxiter)
                              
                repl_lambda_parent = obj.adj_mats{1}*obj.nest_dissim_params{1};
                
                for nbiterations =1:8
                % update the values of mean utilities
                nablaF2mu = sum(gammaiq{1}, 1);                
                beta_sums = sum(betaiq{1}, 1);                                
                a_l = repl_lambda_parent'.* (beta_sums./repl_lambda_parent'-nablaF2mu) + b_l;
                                
                obj.mean_utils = obj.mean_utils + repl_lambda_parent'.*(log(b_l./a_l));

                % update vs, betas, and gammas
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                end
                % shifting
                %obj.mean_utils = 1 + obj.mean_utils - min(obj.mean_utils);
                %v_wts = obj.feedforward(offersets);
                %[gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                %nablaF2mu = sum(gammaiq{1}, 1);
                %beta_sums = sum(betaiq{1}, 1);
                %current_mu = obj.mean_utils;
                %gradient_mu = (beta_sums./repl_lambda_parent'-nablaF2mu);

                %gradient_theta = gradient_mu.*obj.mean_utils;
                    
                    %update values of lambdas through one iteration of GD
                    Z_star = {};
                    nablaF2_list = {};
                    curr_taus = {};
                    for level=2:obj.num_levels
                        curr_taus{level-1} = log(obj.nest_dissim_params{level-1});
                        probs = obj.local_probs(level-1, v_wts);
                        % compute Delta
                        Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
                        tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
                        Delta = Delta_first_term - Delta_second_term;
                        % derivative of F2
                        nablaF2_first_term = gammaiq{level}.*Delta;
                        tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
                        nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
                        nablaF2_list{level-1} = nablaF2;
                        % derivative of F1
                        nablaF1_first_term = gamma1iq{level-1}.*Delta;
                        nablaF1_second_term = bsxfun(@rdivide, node_sales{level}.*log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}'.^2);
                        nablaF1 = sum(nablaF1_first_term - nablaF1_second_term, 1);

                        fw_c = -obj.nest_dissim_params{level-1}'.*(nablaF1 - nablaF2);
                        
                        if(level==2)
                            Z_star{level-1} = fw_c;
                        else
                            Z_star{level-1} = fw_c + (Z_star{level-2})*obj.adj_mats{level-1};
                        end            
                    end
                    
                    gradient_delta = reshape(cell2mat(Z_star)',[],1);

                    % Gradient:
                    %current_gradient = [gradient_theta,gradient_delta',0];
                    current_gradient = [zeros(1,obj.num_prods),gradient_delta',0];
                    tmp_mu = obj.mean_utils+1-min(obj.mean_utils);
             
                    current_lambda = reshape(cell2mat(obj.nest_dissim_params'),[],1);
                    % mu, lambda
                    %before_transform_x = [tmp_mu,current_lambda'];
                    % theta, delta
                    %x = log(before_transform_x)*Box_tran;
                    x = [obj.mean_utils,log(current_lambda')*Box'];

                    % Backtracking Step size
                     alpha = 0.5;
                         
                     % take step:
                     xnew = x - alpha*current_gradient;
                     
                     % projection
                     xnew = max([xnew;lb]);
                    
                    %f_transformed = @(x) obj.pgd_opt_fun_lin(exp(x/Box_tran), before_transform_x, curr_taus, offersets, sales, node_sales, nablaF2_list);
                    %f_transformed = @(x) obj.pgd_opt_fun(exp(x/Box_tran), curr_taus, offersets, sales, node_sales, nablaF2_list);
                    f_transformed = @(x) obj.opt_func([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], offersets, sales);
                    %f_transformed = @(x) obj.opt_func(exp(x/Box_tran), offersets, sales);
                    %Stopping_Condition = f_transformed(xnew)>f_transformed(x)-alpha/2*norm(current_gradient)^2;
                     fval_old = f_transformed(x);
                     
                     ready = false;
                    while ~ready
                        try
                            Stopping_Condition = f_transformed(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                            ready = true;
                        catch ME
                            alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            fprintf('%s\ntrying it again...\n', ME.message);
                        end
                    end
                     
                     %Stopping_Condition = f_transformed(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     
                     while (Stopping_Condition)
                         % x = xnew;
                         alpha = alpha*0.5;
                         xnew = x - alpha*current_gradient;
                         xnew = max([xnew;lb]);
                         %fprintf('alpha %f, %f\n', alpha,f_transformed(xnew));
                         Stopping_Condition = f_transformed(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     end                                      
                     

                         %fprintf('alpha %f, %f, %f\n', alpha(1),alpha(2),f_transformed(xnew));
                     
                    % check step
                    if ~isfinite(xnew)
                        display(['Number of iterations: ' num2str(niter)])
                        error('x is inf or NaN')
                    end
                    %fprintf('niter %f\n', niter);

                    niter = niter + 1;
                    
                        update_xnew = [xnew(1:obj.num_prods),exp(xnew(obj.num_prods+1:length(x))/Box')];
                        %update_xnew = exp(xnew/Box_tran);
                        %obj.mean_utils = current_mu;
                        cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            obj.nest_dissim_params{level} = update_xnew(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end

                        v_wts = obj.feedforward(offersets);
                        [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                    curr_obj = -obj.logLL(offersets, sales);
                    %fprintf('Found step size %f,and the value of original objective %f\n', alpha,curr_obj);
                    %fprintf('%f, %f\n', alpha,curr_obj);
                    %fprintf('the value of original objective %f\n',curr_obj);

                    %dfval = prev_obj - curr_obj;
                    %x = xnew;
                    %prev_obj = curr_obj; 
            end
            
            %display(['Number of iterations: ' num2str(niter)])
            fval = curr_obj;
            fprintf('Fullback NLL 1d Algo. termininated in %d iters...obj val %f\n', niter, fval);
        end % mm on mu, pgd on delta backtracking 1d, F1-F2 005
   

        function xnew = projected_backtracking_linesearch(obj,current_gradient,lb,x,f,Box)
               % Initial step size
                    alpha = 0.5;

                    % take step:
                     xnew = x - alpha*current_gradient;

                     % projection
                     xnew = max([xnew;lb]);
                     fval_old = f(x);
                     while(any(exp(xnew(obj.num_prods+1:length(x))/Box')<1e-3))                            
                                alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            %fprintf('alpha %f\n', alpha*10000);
                     end

                    ready = false;
                    while ~ready
                        try
                            Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                            ready = true;
                        catch ME
                            alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            fprintf('%s\ntrying it again...\n', ME.message);
                        end
                    end
                    %Stopping_Condition = f_transformed(xnew)>f_transformed(x)-alpha/2*norm(current_gradient)^2;
                    % Stopping_Condition = f(xnew)>f(x)+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     
                     while (Stopping_Condition) 
                         % x = xnew;
                         alpha = alpha/2;
                         xnew = x - alpha*current_gradient;
                         xnew = max([xnew;lb]);
                        
                         ready = false;
                        while ~ready
                            try
                                Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                                ready = true;
                            catch ME
                                alpha = alpha*0.5;
                                xnew = x - alpha*current_gradient;
                                xnew = max([xnew;lb]);
                                fprintf('%s\ntrying it again...\n', ME.message);
                            end
                        end
                         %Stopping_Condition = f(xnew)>f(x)+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     end
                    
                         %fprintf('alpha %f, %f\n', alpha,improvement);
                        
                     %fprintf('Found step size %f, obj val %f, and number of iterations %f\n', alpha,f(xnew), nbiter);
                    
        end

        function fval = fit_agd_mu_delta(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            % uses gd algo to estimate the parameters of the model
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            % Box transformation matrix
            Box = -eye(num_non_leaf_nodes);
            Box_i = 0;
            for level=2:obj.num_levels
                for node=1:obj.num_nodes_per_level(level)
                    Box_i = Box_i+1;
                    Box_pr = obj.adj_mats{level};
                    Box_j = sum(obj.num_nodes_per_level(2:level))+find(Box_pr(node,:));
                    Box(Box_i,Box_j) = 1;
                end
            end
            
            Box_tran = [eye(obj.num_prods),zeros(obj.num_prods,num_non_leaf_nodes);zeros(num_non_leaf_nodes,obj.num_prods),Box]';
           
            
            
            % x variable: [mu, delata...]
            
            % Box constraints
            lb = [(-Inf)*ones(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];
            
            % Initial value
            x0 = [zeros(1, obj.num_prods), zeros(1,num_non_leaf_nodes-1),0];

            
            % GD interation options
            %tol = 1e-16;
            maxiter = 299;
            
            % initialize 
            x_curr = x0;
            
            niter = 0;
            %dfval = Inf;
           
            
            % compute sales at each node in the tree and initialize the
                % parameters of the model
                obj.mean_utils = ones(1, obj.num_prods);
                node_sales = {sales};
                for level=1:obj.num_levels
                    node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                    obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
                end
                obj.nest_dissim_params{obj.num_levels} = 1;
                        
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);


            curr_obj = -obj.logLL(offersets, sales);
            curr_time = toc;
            fprintf(2,'AGD neglog at iter 0 is %f with time %f\n', curr_obj, curr_time);
            % Gradient descent
            while (niter <= maxiter)                         
                %fprintf('starting iter %d\n', niter);
                nablaF2mu = sum(gammaiq{1}, 1);
                repl_lambda_parent = obj.adj_mats{1}*obj.nest_dissim_params{1};
                beta_sums = sum(betaiq{1}, 1);

                gradient_mu = (beta_sums./repl_lambda_parent'-nablaF2mu);
                current_lambda = reshape(cell2mat(obj.nest_dissim_params'),[],1);
                current_mu = obj.mean_utils;
                x_curr = [current_mu,log(current_lambda')*Box'];

                % DO a mu update
                current_gradient = [gradient_mu,zeros(1,num_non_leaf_nodes-1),0];
                % Define the objective function on theta and delta
                f = @(x) obj.opt_func([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], offersets, sales);
                % do pgd to get update mu
                xnew = obj.projected_backtracking_linesearch(current_gradient,lb,x_curr,f,Box);
                    % check step
                if ~isfinite(xnew)
                    display(['Number of iterations: ' num2str(niter)])
                    error('x is inf or NaN')
                end
                update_xnew = [xnew(1:obj.num_prods),exp(xnew(obj.num_prods+1:length(x_curr))/Box')];
                % UPDATE estimate of mu
                obj.mean_utils = update_xnew(1:obj.num_prods);
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                current_mu = obj.mean_utils;

                %gradient_theta = gradient_mu.*current_mu;
                    
                    %update values of lambdas through one iteration of FW
                    Z_star = {};
                    nablaF2_list = {};
                    curr_taus = {};
                    for level=2:obj.num_levels
                        curr_taus{level-1} = log(obj.nest_dissim_params{level-1});
                        probs = obj.local_probs(level-1, v_wts);
                        % compute Delta
                        Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
                        tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
                        Delta = Delta_first_term - Delta_second_term;
                        % derivative of F2
                        nablaF2_first_term = gammaiq{level}.*Delta;
                        tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
                        nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
                        nablaF2_list{level-1} = nablaF2;
                        % derivative of F1
                        nablaF1_first_term = gamma1iq{level-1}.*Delta;
                        nablaF1_second_term = bsxfun(@rdivide, node_sales{level}.*log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}'.^2);
                        nablaF1 = sum(nablaF1_first_term - nablaF1_second_term, 1);

                        fw_c = -obj.nest_dissim_params{level-1}'.*(nablaF1 - nablaF2);
                        
                        if(level==2)
                            Z_star{level-1} = fw_c;
                        else
                            Z_star{level-1} = fw_c + (Z_star{level-2})*obj.adj_mats{level-1};
                        end            
                    end
                    
                    gradient_delta = reshape(cell2mat(Z_star)',[],1);
                    
                    current_lambda = reshape(cell2mat(obj.nest_dissim_params'),[],1);
                    % mu, lambda 
                    % theta, delta
                    %x = log(before_transform_x)*Box_tran;
                    x_curr = [current_mu,log(current_lambda')*Box'];
                    % Gradient:
                    current_gradient = [zeros(1,obj.num_prods),gradient_delta',0];
                    % Define the objective function on theta and delta
                    f = @(x) obj.pgd_opt_fun_lin_delta([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], curr_taus, offersets, sales, node_sales, nablaF2_list);
                    xnew = obj.projected_backtracking_linesearch(current_gradient,lb,x_curr,f,Box);
                      
                    % check step
                    if ~isfinite(xnew)
                        display(['Number of iterations: ' num2str(niter)])
                        error('x is inf or NaN')
                    end
                    %fprintf('niter %f\n', niter);
                    % update termination
                    % metrics
                    %dfval = norm(f(x)-f(xnew));
                    
                        update_xnew = [xnew(1:obj.num_prods),exp(xnew(obj.num_prods+1:length(x_curr))/Box')];
                        cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            obj.nest_dissim_params{level} = update_xnew(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end

                        v_wts = obj.feedforward(offersets);
                        [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                        curr_obj = -obj.logLL(offersets, sales);
                        niter = niter + 1;


                        curr_obj = -obj.logLL(offersets, sales);
                        curr_time = toc;
                        fprintf(2,'AGD neglog at iter %d is %f with time %f\n', niter, curr_obj, curr_time);


                    %x = xnew;
            end

            %display(['Number of iterations: ' num2str(niter)])
            fval = curr_obj;
            fprintf('AGD backtracking Algo. termininated in %d iters...obj val %f\n', niter, fval);
        end  %gd backtracking 1d NLL mu delta




        function fval = fit_gdnll_mudelta(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            % uses gd algo to estimate the parameters of the model
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            % Box transformation matrix
            Box = -eye(num_non_leaf_nodes);
            Box_i = 0;
            for level=2:obj.num_levels
                for node=1:obj.num_nodes_per_level(level)
                    Box_i = Box_i+1;
                    Box_pr = obj.adj_mats{level};
                    Box_j = sum(obj.num_nodes_per_level(2:level))+find(Box_pr(node,:));
                    Box(Box_i,Box_j) = 1;
                end
            end
            
            Box_tran = [eye(obj.num_prods),zeros(obj.num_prods,num_non_leaf_nodes);zeros(num_non_leaf_nodes,obj.num_prods),Box]';
           
            
            
            % x variable: [mu, delata...]
            
            % Box constraints
            lb = [(-Inf)*ones(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];
            
            % Initial value
            x0 = [zeros(1, obj.num_prods), zeros(1,num_non_leaf_nodes-1),0];

            
            % GD interation options
            %tol = 1e-16;
            maxiter = 199;
            
            % initialize 
            x = x0;
            
            niter = 0;
            %dfval = Inf;
           
            
            % compute sales at each node in the tree and initialize the
                % parameters of the model
                obj.mean_utils = ones(1, obj.num_prods);
                node_sales = {sales};
                for level=1:obj.num_levels
                    node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                    obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
                end
                obj.nest_dissim_params{obj.num_levels} = 1;
                        
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                
            % Gradient descent
            while (niter <= maxiter)                         
                nablaF2mu = sum(gammaiq{1}, 1);
                repl_lambda_parent = obj.adj_mats{1}*obj.nest_dissim_params{1};
                beta_sums = sum(betaiq{1}, 1);

                gradient_mu = (beta_sums./repl_lambda_parent'-nablaF2mu);

                current_mu = obj.mean_utils;

                %gradient_theta = gradient_mu.*current_mu;
                    
                    %update values of lambdas through one iteration of FW
                    Z_star = {};
                    nablaF2_list = {};
                    curr_taus = {};
                    for level=2:obj.num_levels
                        curr_taus{level-1} = log(obj.nest_dissim_params{level-1});
                        probs = obj.local_probs(level-1, v_wts);
                        % compute Delta
                        Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
                        tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
                        Delta = Delta_first_term - Delta_second_term;
                        % derivative of F2
                        nablaF2_first_term = gammaiq{level}.*Delta;
                        tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
                        nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
                        nablaF2_list{level-1} = nablaF2;
                        % derivative of F1
                        nablaF1_first_term = gamma1iq{level-1}.*Delta;
                        nablaF1_second_term = bsxfun(@rdivide, node_sales{level}.*log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}'.^2);
                        nablaF1 = sum(nablaF1_first_term - nablaF1_second_term, 1);

                        fw_c = -obj.nest_dissim_params{level-1}'.*(nablaF1 - nablaF2);
                        
                        if(level==2)
                            Z_star{level-1} = fw_c;
                        else
                            Z_star{level-1} = fw_c + (Z_star{level-2})*obj.adj_mats{level-1};
                        end            
                    end
                    
                    gradient_delta = reshape(cell2mat(Z_star)',[],1);
                    
                    current_lambda = reshape(cell2mat(obj.nest_dissim_params'),[],1);
                    % mu, lambda
                    before_transform_x = [current_mu,current_lambda']; 
                    % theta, delta
                    %x = log(before_transform_x)*Box_tran;
                    x = [current_mu,log(current_lambda')*Box'];
                    % Gradient:
                    current_gradient = [gradient_mu,gradient_delta',0];
                    
                     % Initial step size
                    alpha = 0.5;

                    % take step:
                     xnew = x - alpha*current_gradient;

                     % projection
                     xnew = max([xnew;lb]);
                     
                     % Define the objective function on theta and delta
                     f = @(x) obj.opt_func([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], offersets, sales);
                    
                     fval_old = f(x);
                    ready = false;
                    while ~ready
                        try
                            Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                            ready = true;
                        catch ME
                            alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            fprintf('%s\ntrying it again...\n', ME.message);
                        end
                    end
                    %Stopping_Condition = f_transformed(xnew)>f_transformed(x)-alpha/2*norm(current_gradient)^2;
                    % Stopping_Condition = f(xnew)>f(x)+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     
                     while (Stopping_Condition) 
                         % x = xnew;
                         alpha = alpha/2;
                         xnew = x - alpha*current_gradient;
                         xnew = max([xnew;lb]);
                        
                         ready = false;
                        while ~ready
                            try
                                Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                                ready = true;
                            catch ME
                                alpha = alpha*0.5;
                                xnew = x - alpha*current_gradient;
                                xnew = max([xnew;lb]);
                                fprintf('%s\ntrying it again...\n', ME.message);
                            end
                        end
                         %Stopping_Condition = f(xnew)>f(x)+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     end
                    

                         %fprintf('alpha %f, %f\n', alpha,improvement);
                        
                     %fprintf('Found step size %f, obj val %f, and number of iterations %f\n', alpha,f(xnew), nbiter);
                    
                    
                    % check step
                    if ~isfinite(xnew)
                        display(['Number of iterations: ' num2str(niter)])
                        error('x is inf or NaN')
                    end
                    %fprintf('niter %f\n', niter);
                    % update termination
                    % metrics
                    niter = niter + 1;
                    %dfval = norm(f(x)-f(xnew));
                    
                        update_xnew = [xnew(1:obj.num_prods),exp(xnew(obj.num_prods+1:length(x))/Box')];
                        obj.mean_utils = update_xnew(1:obj.num_prods);
                        cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            obj.nest_dissim_params{level} = update_xnew(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end

                        v_wts = obj.feedforward(offersets);
                        [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                        curr_obj = -obj.logLL(offersets, sales);
                    
                    %x = xnew;
            end

            %display(['Number of iterations: ' num2str(niter)])
            fval = curr_obj;
            fprintf('GD backtracking Algo. termininated in %d iters...obj val %f\n', niter, fval);
        end  %gd backtracking 1d NLL mu delta
             
        function fval = fit_gdlin_mudelta(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            % uses gd algo to estimate the parameters of the model
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            % Box transformation matrix
            Box = -eye(num_non_leaf_nodes);
            Box_i = 0;
            for level=2:obj.num_levels
                for node=1:obj.num_nodes_per_level(level)
                    Box_i = Box_i+1;
                    Box_pr = obj.adj_mats{level};
                    Box_j = sum(obj.num_nodes_per_level(2:level))+find(Box_pr(node,:));
                    Box(Box_i,Box_j) = 1;
                end
            end
            
            Box_tran = [eye(obj.num_prods),zeros(obj.num_prods,num_non_leaf_nodes);zeros(num_non_leaf_nodes,obj.num_prods),Box]';
                       
            % x variable: [theta, delata...]
            
            % Box constraints
            lb = [(-Inf)*ones(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];
            
            % Initial value
            x0 = [zeros(1, obj.num_prods), zeros(1,num_non_leaf_nodes-1),0];

            
            % GD interation options
            %tol = 1e-16;
            maxiter = 199;
            
            % initialize 
            x = x0;
            
            niter = 0;
            %dfval = Inf;
           
            
            % compute sales at each node in the tree and initialize the
                % parameters of the model
                obj.mean_utils = ones(1, obj.num_prods);
                node_sales = {sales};
                for level=1:obj.num_levels
                    node_sales{level+1} = node_sales{level}*obj.adj_mats{level};
                    obj.nest_dissim_params{level} = ones(obj.num_nodes_per_level(level+1), 1);
                end
                obj.nest_dissim_params{obj.num_levels} = 1;
                        
                v_wts = obj.feedforward(offersets);
                [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                
            % Gradient descent
            while (niter <= maxiter)                         
                nablaF2mu = sum(gammaiq{1}, 1);
                repl_lambda_parent = obj.adj_mats{1}*obj.nest_dissim_params{1};
                beta_sums = sum(betaiq{1}, 1);

                gradient_mu = (beta_sums./repl_lambda_parent'-nablaF2mu);

                current_mu = obj.mean_utils;

                %gradient_theta = gradient_mu.*current_mu;
                    
                    %update values of lambdas through one iteration of FW
                    Z_star = {};
                    nablaF2_list = {};
                    curr_taus = {};
                    for level=2:obj.num_levels
                        curr_taus{level-1} = log(obj.nest_dissim_params{level-1});
                        probs = obj.local_probs(level-1, v_wts);
                        % compute Delta
                        Delta_first_term = bsxfun(@rdivide, log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}');
                        tmp_Delta_second_term = (probs.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        Delta_second_term = bsxfun(@rdivide, tmp_Delta_second_term, obj.nest_dissim_params{level-1}');
                        Delta = Delta_first_term - Delta_second_term;
                        % derivative of F2
                        nablaF2_first_term = gammaiq{level}.*Delta;
                        tmp_nablaF2_second_term = (node_sales{level-1}.*log(v_wts{level-1} + (v_wts{level-1} == 0)))*obj.adj_mats{level-1};
                        nabla_second_term = bsxfun(@rdivide, tmp_nablaF2_second_term, obj.nest_dissim_params{level-1}'.^2);
                        nablaF2 = sum(nablaF2_first_term - nabla_second_term, 1);
                        nablaF2_list{level-1} = nablaF2;
                        % derivative of F1
                        nablaF1_first_term = gamma1iq{level-1}.*Delta;
                        nablaF1_second_term = bsxfun(@rdivide, node_sales{level}.*log(v_wts{level} + (v_wts{level} == 0)), obj.nest_dissim_params{level-1}'.^2);
                        nablaF1 = sum(nablaF1_first_term - nablaF1_second_term, 1);

                        fw_c = -obj.nest_dissim_params{level-1}'.*(nablaF1 - nablaF2);
                        
                        if(level==2)
                            Z_star{level-1} = fw_c;
                        else
                            Z_star{level-1} = fw_c + (Z_star{level-2})*obj.adj_mats{level-1};
                        end            
                    end
                    
                    gradient_delta = reshape(cell2mat(Z_star)',[],1);
                    
                    current_lambda = reshape(cell2mat(obj.nest_dissim_params'),[],1);
                    % mu, lambda
                    before_transform_x = [current_mu,current_lambda']; 
                    % theta, delta
                    %x = log(before_transform_x)*Box_tran;
                    x = [current_mu,log(current_lambda')*Box'];
                    % Gradient:
                    current_gradient = [gradient_mu,gradient_delta',0];
                    
                     % Initial step size
                    alpha = 0.5;

                    % take step:
                     xnew = x - alpha*current_gradient;

                     % projection
                     xnew = max([xnew;lb]);
                     
                     % Define the objective function on theta and delta
                     %f = @(x) obj.opt_func([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], offersets, sales);
                     f = @(x) obj.pgd_all_opt_fun_lin([x(1:obj.num_prods),exp(x(obj.num_prods+1:length(x))/Box')], before_transform_x, curr_taus, current_mu, offersets, sales, node_sales, nablaF2mu, nablaF2_list);
                    
                     fval_old = f(x);
                     
                     while(any(exp(xnew(obj.num_prods+1:length(x))/Box')<1e-3))                            
                                alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            %fprintf('alpha %f\n', alpha*10000);
                     end
                     
                    ready = false;
                    while ~ready
                        try
                            Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;                         
                            ready = true;
                        catch ME
                            alpha = alpha*0.5;
                            xnew = x - alpha*current_gradient;
                            xnew = max([xnew;lb]);
                            fprintf('%s\ntrying it again...\n', ME.message);
                        end
                    end
                    %Stopping_Condition = f_transformed(xnew)>f_transformed(x)-alpha/2*norm(current_gradient)^2;
                    % Stopping_Condition = f(xnew)>f(x)+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     
                     while (Stopping_Condition) 
                         % x = xnew;
                         alpha = alpha/2;
                         xnew = x - alpha*current_gradient;
                         xnew = max([xnew;lb]);
                        
                         ready = false;
                        while ~ready
                            try
                                Stopping_Condition = f(xnew)>fval_old+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                                ready = true;
                            catch ME
                                alpha = alpha*0.5;
                                xnew = x - alpha*current_gradient;
                                xnew = max([xnew;lb]);
                                fprintf('%s\ntrying it again...\n', ME.message);
                            end
                        end
                         %Stopping_Condition = f(xnew)>f(x)+current_gradient*((xnew-x)')+1/(2*alpha)*norm(xnew-x)^2;
                     end
                    

                         %fprintf('alpha %f, %f\n', alpha,improvement);
                        
                     %fprintf('Found step size %f, obj val %f\n', alpha,f(xnew));
                    
                    
                    % check step
                    if ~isfinite(xnew)
                        display(['Number of iterations: ' num2str(niter)])
                        error('x is inf or NaN')
                    end
                    %fprintf('niter %f\n', niter);
                    % update termination
                    % metrics
                    niter = niter + 1;
                    %dfval = norm(f(x)-f(xnew));
                    
                        update_xnew = [xnew(1:obj.num_prods),exp(xnew(obj.num_prods+1:length(x))/Box')];
                        obj.mean_utils = update_xnew(1:obj.num_prods);
                        cum_sum_nodes_per_level = cumsum(obj.num_nodes_per_level);
                        for level=1:obj.num_levels
                            obj.nest_dissim_params{level} = update_xnew(cum_sum_nodes_per_level(level)+1:cum_sum_nodes_per_level(level+1))';
                        end

                        v_wts = obj.feedforward(offersets);
                        [gamma1iq, gammaiq, betaiq] = obj.update_gamma1_gamma_beta(node_sales, v_wts);
                        curr_obj = -obj.logLL(offersets, sales);
                    
                    %x = xnew;
            end

            %display(['Number of iterations: ' num2str(niter)])
            fval = curr_obj;
            fprintf('GD backtracking Algo. termininated in %d iters...obj val %f\n', niter, fval);
        end  %gd backtracking 1d Lin mu delta
        
    end 
end

