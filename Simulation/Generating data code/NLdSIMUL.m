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
            defaultNITERS = 2000;
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
        
        
%         function choice_probs = compute_proba(obj, offersets)
%             % uses random walk to compute choice probabilitdddddies
%             v_wts = obj.feedforward(offersets);
%             num_offersets = size(offersets, 1);
%             probs = ones(num_offersets, 1); % initialize the probability of choosing the root node to 1
%             for level=obj.num_levels:-1:1
%                 total_wts = v_wts{level}*obj.adj_mats{level};
%                 repl_total_wts = total_wts*obj.adj_mats{level}'; % replicate wts so that it is easy to divide
%                 repl_probs = probs*obj.adj_mats{level}'; % replicate probs of entering the nodes
%                 probs = (v_wts{level}./(repl_total_wts + (repl_total_wts==0))).*repl_probs;
%             end
%             assert(all(round(abs(sum(probs, 2) - 1), 7) == 0));
%             choice_probs = probs;
%         end
    
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
        end

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
%                 op = [fval, exitflag];
            elseif(strcmp(method, 'mm'))
                op = obj.fit_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'knitro'))
                op = obj.fit_knitro(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'knitrotran'))
                op = obj.fit_knitrotran(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'knitrobox'))
                op = obj.fit_knitrobox(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'fw_mm'))
                op = obj.fit_fw_mm(offersets, sales, mu_lb, mu_ub, lambda_lb);
            elseif(strcmp(method, 'gd'))
                op = obj.fit_gd(offersets, sales, mu_lb, mu_ub, lambda_lb);
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
        
        
        function curr_obj = fit_mm(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
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
            N_ITERS = 10000;
            num_offersets = size(offersets, 1);
            v_wts = obj.feedforward(offersets);
            [gammaiq, betaiq] = obj.update_gamma_beta(node_sales, v_wts);
            
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
                [gammaiq, betaiq] = obj.update_gamma_beta(node_sales, v_wts);
                curr_obj = -obj.logLL(offersets, sales);
%                 assert(round(prev_obj - curr_obj, 7) >= 0);
                if(~(round(prev_obj - curr_obj, 7) >= 0))
                    fprintf('%d,%f\n',n_iter,prev_obj-curr_obj);
                    break;
                end

                
                % update the values of lambdas
%                 for level=1:obj.num_levels-1
                for level=obj.num_levels-1:-1:1
                    parent_lambdas = obj.adj_mats{level+1}*obj.nest_dissim_params{level+1};
                    if(level == 1)
                        child_lambdas = zeros(obj.num_nodes_per_level(level+1), 1);
                    else
                        child_lambdas = max(bsxfun(@times, obj.adj_mats{level}, obj.nest_dissim_params{level-1}))';
                    end
                    tmp_lambda = obj.opt_lambda(level+1, v_wts, gammaiq, betaiq, node_sales);
%                     obj.nest_dissim_params{level} = max(min(tmp_lambda', parent_lambdas), child_lambdas);
                    assert(all(tmp_lambda' <= parent_lambdas));
                    obj.nest_dissim_params{level} = tmp_lambda';
                    
%                     for i=1:obj.num_nodes_per_level(level+1)
%                         obj.nest_dissim_params{level}(i) = max(min(tmp_lambda(i), parent_lambdas(i)), child_lambdas(i));
%                     end
                    % update the relevant quantities
                    v_wts = obj.feedforward(offersets);
                    [gammaiq, betaiq] = obj.update_gamma_beta(node_sales, v_wts);
                    curr_obj = -obj.logLL(offersets, sales);
%                     assert(curr_obj <= prev_obj);                
                end

%                 curr_obj = -obj.logLL(offersets, sales);
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
        
        function fval = fit_fmincon(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
            f = @(x) obj.opt_func(x, offersets, sales);
            
            % x variable: [mean_utils, dissim_param_level_1, ...,
            % dissim_param_level_d]
            num_non_leaf_nodes = sum(obj.num_nodes_per_level) - obj.num_prods;
            lb = [mu_lb*ones(1,obj.num_prods), lambda_lb*ones(1,num_non_leaf_nodes-1), 1];
            ub = [mu_ub*ones(1,obj.num_prods), ones(1,num_non_leaf_nodes)];
            lb(1) = 0;
            ub(1) = 0;
            options = optimoptions('fmincon', 'Display', 'off');
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
            [x, fval, exitflag, output] = fmincon(f, x0, Aleq, bleq, [], [], lb, ub, [], options);            
            % make sure that the lambda constraints are satisfied
            for level=1:obj.num_levels-1
                curr_lambdas = obj.nest_dissim_params{level};
                parent_lambdas = obj.adj_mats{level+1}*obj.nest_dissim_params{level+1};
                if(all(curr_lambdas <= parent_lambdas) == 0)
                   fprintf('Solution obtained by fmincon infeasible...\n'); 
                end
%                 assert(all(curr_lambdas <= parent_lambdas));
            end
            %[x, fval, exitflag, output] = fmincon(f, zeros(1,obj.num_prods+obj.num_nests), [], [], [], [], lb, ub, []);            
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
            options = knitro_options('maxit', 2000, 'maxtime_real', 6e2, 'xtol', 1e-5);
            [x, fval, exitflag, output, lambda, grad, hessian] = knitro_nlp(f, x0, Aleq, bleq, [], [], lb, ub, [], [],options);            
            % make sure that the lambda constraints are satisfied
            for level=1:obj.num_levels-1
                curr_lambdas = obj.nest_dissim_params{level};
                parent_lambdas = obj.adj_mats{level+1}*obj.nest_dissim_params{level+1};
                if(all(curr_lambdas <= parent_lambdas) == 0)
                   fprintf('Solution obtained by fmincon infeasible...\n'); 
                end
%                 assert(all(curr_lambdas <= parent_lambdas));
            end
            %[x, fval, exitflag, output] = fmincon(f, zeros(1,obj.num_prods+obj.num_nests), [], [], [], [], lb, ub, []);            
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
        
        function fval = fit_gd(obj, offersets, sales, mu_lb, mu_ub, lambda_lb)
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
           
            % Define the objective function on theta and delta
            f = @(x) obj.opt_func(exp(x/Box_tran), offersets, sales);
            
            % x variable: [theta, delata...]
            
            % Box constraints
            lb = [zeros(1,obj.num_prods), zeros(1,num_non_leaf_nodes)];
            ub = [log(2+exp(1))*ones(1,obj.num_prods), (-log(lambda_lb))*ones(1,num_non_leaf_nodes-1),0];
            lb(1) = 1;
            ub(1) = 1;
            
            % Initial value
            x0 = [ones(1, obj.num_prods), zeros(1,num_non_leaf_nodes-1),0];
            %x0 = 30/100*ub+70/100*lb;
            
            % GD interation options
            tol = 1e-16;
            maxiter = 2000;
            
            % Initial step size
            alpha = 0.01;
            
            % initialize 
            x = x0;
            
            niter = 0;
            dfval = Inf;
           
            
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
            while and(niter <= maxiter, dfval >= tol)
                % calculate gradient:
                %g = Grad(f, x);   
                          
                
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
                %curr_obj = -obj.logLL(offersets, sales);
%                 assert(round(prev_obj - curr_obj, 7) >= 0);
                
                gradient_mu = (beta_sums./repl_lambda_parent'-nablaF2mu);
                backward_tran_x = exp(x/Box_tran);
                current_mu = backward_tran_x(1:length(gradient_mu));
                %lambda = delta(length(g1)+1:end);
                gradient_theta = gradient_mu.*current_mu;
                    
                    
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
                            Z_star{level-1} = -fw_c;
                        else
                            Z_star{level-1} = -fw_c - (Z_star{level-2})*obj.adj_mats{level-1};
                        end            
                    end
                    
                    gradient_delta = reshape(cell2mat(Z_star)',[],1);
                    
                    % Gradient:
                    current_gradient = [gradient_theta,gradient_delta',0];
                    % the first element is always 0
                    current_gradient(1) = 0;
                    % Step size
                    function_step = @(k) f(x-k*current_gradient);
                    
                    options = optimoptions('fmincon', 'Display', 'off');
                    %options = optimoptions('fmincon', 'Display', 'notify-detailed');
                    options.StepTolerance = 1e-20;
                    options.ConstraintTolerance = 1e-4;
                    options.MaxIterations = 5000;
                    
                    %A_step = [-current_gradient';current_gradient'];
                    %b_step = [(-x+ub)';(x-lb)'];
                    
                    
                    %upper_bound = (-x+ub)./(-current_gradient);
                    %lower_bound = (x-lb)./(current_gradient);
                    
                    %lower_bound_positive = (x(current_gradient>0)-ub(current_gradient>0))./(current_gradient(current_gradient>0));
                    %lower_bound_negative = (x(current_gradient<0)-lb(current_gradient<0))./(current_gradient(current_gradient<0));
                    
                    %lower_bound = max([lower_bound_positive,lower_bound_negative]);
                    
                    %upper_bound_positive = (x(current_gradient>0)-lb(current_gradient>0))./(current_gradient(current_gradient>0));
                    %upper_bound_negative = (x(current_gradient<0)-ub(current_gradient<0))./(current_gradient(current_gradient<0));
                    
                    %upper_bound = min([upper_bound_positive,upper_bound_negative]);
                    
                    step_size_denominator = max(abs(current_gradient))/2;
                    
                    %[k, fval, exitflag, output] = fmincon(function_step, 0, A_step, b_step, [], [], [], [], [], options); 
                    %[k, fval, exitflag, output] = fmincon(function_step, 0, [], [], [], [], lower_bound, upper_bound,[],options); 
                    [k, fval, exitflag, output] = fmincon(function_step, 0, [], [], [], [], -1/step_size_denominator, 1/step_size_denominator,[],options);
                    %[k, fval, exitflag, output] = fminunc(function_step, 0); 
                                        
                    alpha = k;
                    %fprintf('step size %f\n', alpha);
                    %fprintf('step size fval %f\n', fval);
                                        
                    % take step:
                    xnew = x - alpha*current_gradient;
                    
                    %projection
                    xnew = min([xnew;ub]);
                    xnew = max([xnew;lb]);
                                        
                    % check step
                    if ~isfinite(xnew)
                        display(['Number of iterations: ' num2str(niter)])
                        error('x is inf or NaN')
                    end
                    %fprintf('niter %f\n', niter);
                    % update termination
                    % metrics
                    niter = niter + 1;
                    dfval = norm(f(x)-f(xnew));
                    
                    x = xnew;
            end
            % make sure that the lambda constraints are satisfied
            
            
            if(all(x>=lb)==0)
                %fval = 100000;
                fprintf('infeasible...\n'); 
            end
            display(['Number of iterations: ' num2str(niter)])
            fval = f(x);
        end
    end
end

