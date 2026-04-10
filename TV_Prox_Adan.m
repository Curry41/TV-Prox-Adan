% Main function: CP_SGD_proximal_Adan_double_with_linear_trans
function [Z, varargout] = TV_Prox_Adan(initial_A, params, params_beta, params_beta_tv, Y_tensor, gen_mask, beta, num_iters, original_tensor, is_add_TV, is_fine_tune, trans_type, is_eva, is_measure_time)

    tensor_shape = size(Y_tensor);
    N_dims = length(tensor_shape);
    
    disp(params); % Display struct fields

    F_matrices = cell(1, N_dims);
    for i = 1:N_dims
        F_matrices{i} = finite_difference_matrix(tensor_shape(i)); % MODIFIED CALL
    end

    % Hyperparameters from structs
    lambda_tv = params.lambda_tv;
    lambda_cp = params.lambda_cp;
    lambda_3 = params.lambda_3;
    
    eta = params.eta;
    Adan_beta1 = params_beta.adan_beta_1;
    Adan_beta2 = params_beta.adan_beta_2;
    Adan_beta3 = params_beta.adan_beta_3;
    epsilon_val = 1e-8; 

    Adan_beta1_tv = params_beta_tv.adan_beta1_tv;
    Adan_beta2_tv = params_beta_tv.adan_beta2_tv;
    Adan_beta3_tv = params_beta_tv.adan_beta3_tv;

    A = initial_A; 

    m = cell(1, N_dims);
    v = cell(1, N_dims);
    n_t_adan = cell(1, N_dims); 
    prev_gradient = cell(1, N_dims);

    for n_idx = 1:N_dims
        m{n_idx} = zeros(size(A{n_idx}));
        v{n_idx} = zeros(size(A{n_idx}));
        n_t_adan{n_idx} = zeros(size(A{n_idx}));
        prev_gradient{n_idx} = zeros(size(A{n_idx}));
    end

    m_tv = cell(1, N_dims);
    v_tv = cell(1, N_dims);
    n_tv_adan = cell(1, N_dims); 
    prev_gradient_tv = cell(1, N_dims);
    for n_idx = 1:N_dims
        m_tv{n_idx} = zeros(size(A{n_idx}));
        v_tv{n_idx} = zeros(size(A{n_idx}));
        n_tv_adan{n_idx} = zeros(size(A{n_idx}));
        prev_gradient_tv{n_idx} = zeros(size(A{n_idx}));
    end
    
    Y_mat = cell(1, N_dims);
    mask_mat = cell(1, N_dims);

    for n_idx = 1:N_dims
        Y_mat{n_idx} = double(my_unfold(Y_tensor, n_idx));
        mask_mat{n_idx} = double(my_unfold(gen_mask, n_idx));
    end

    loss_history = []; 
    PSNR_history = [];
    SSIM_history = [];
    RSE_history = [];
    TV_history = [];
    Grad_TV_history = [];
    

    for it = 1:num_iters

        % if it>1 && check_factor_convergence(A_pre, A)
        %     fprintf('Converged at iteration %d\n', it); % <- Print message
        %     break;
        % end

        if is_measure_time 
            fprintf('Iteration %d\n', it-1);
        end
        % fprintf('Iteration %d\n', it);
        tv_sum = 0;

        if mod(it, 30) == 0 && it > 100 
            eta = eta * 0.5;
        end

        grad_cp_sum = 0;
        grad_tv_sum = 0;
        
        for n = 1:N_dims 
            
            % fprintf("mode: %d\n", n);

            
            A_kr = khatrirao_except_n(A, n);
            
          
            gradient_val = lambda_cp * mask_mat{n} .* (A{n} * A_kr' - Y_mat{n}) * A_kr;

            
            if is_eva
                grad_cp = sum(abs(gradient_val(:)));
                grad_cp_sum = grad_cp_sum + grad_cp;
            end

            m{n} = (1 - Adan_beta1) * m{n} + Adan_beta1 * gradient_val;
            v{n} = (1 - Adan_beta2) * v{n} + Adan_beta2 * (gradient_val - prev_gradient{n});
            n_t_adan{n} = (1 - Adan_beta3) * n_t_adan{n} + Adan_beta3 * (gradient_val + (1 - Adan_beta2) * (gradient_val - prev_gradient{n})) .^ 2;

            eta_k = eta ./ (sqrt(n_t_adan{n}) + epsilon_val); 

            A{n} = (1 / (1 + lambda_3 * eta)) * (A{n} - eta_k .* (m{n} + (1 - Adan_beta2) * v{n}));
                      

            if (is_add_TV && beta(n) > 0)
                    F_n = F_matrices{n};
                    TV_term = F_n * (A{n} * A_kr');
                    
                    base_tv_grad = beta(n) * lambda_tv * (F_n' * sign(TV_term) * A_kr);
                    gradient_tv = base_tv_grad;

                           
                    if is_eva
                        tv_grad = sum(abs(gradient_tv(:)));
                        grad_tv_sum = grad_tv_sum + tv_grad;
                    end

                    m_tv{n} = (1 - Adan_beta1_tv) * m_tv{n} + Adan_beta1_tv * gradient_tv;
                    v_tv{n} = (1 - Adan_beta2_tv) * v_tv{n} + Adan_beta2_tv * (gradient_tv - prev_gradient_tv{n});
                    n_tv_adan{n} = (1 - Adan_beta3_tv) * n_tv_adan{n} + Adan_beta3_tv * (gradient_tv + (1 - Adan_beta2_tv) * (gradient_tv - prev_gradient_tv{n})) .^ 2;

                    eta_k_tv = eta ./ (sqrt(n_tv_adan{n}) + epsilon_val);
                    
                    A{n} = (1 / (1 + lambda_3 * eta)) * (A{n} - eta_k_tv .* (m_tv{n} + (1 - Adan_beta2_tv) * v_tv{n}));
                    
                    prev_gradient_tv{n} = gradient_tv;

                    
                    if is_eva && beta(n) == 1 && exist('F_n','var') 
                        tv_sum = tv_sum + sum(abs(F_n * (A{n} * A_kr'))); 
                    end
            end
            prev_gradient{n} = gradient_val; 
        end

        if is_eva
            TV_history = [TV_history, tv_sum];
            % Grad_CP_history = [Grad_CP_history, grad_cp_sum];
            Grad_TV_history = [Grad_TV_history, grad_tv_sum];

            Z_temp = reconstruct_tensor(A); % MODIFIED CALL
            [PSNR_value, SSIM_value, RSE_value] = compute_tensor_metrics(Z_temp, original_tensor);
            PSNR_history = [PSNR_history, PSNR_value]; % MODIFIED CALL
            SSIM_history = [SSIM_history, SSIM_value]; % MODIFIED CALL
            RSE_history = [RSE_history, RSE_value]; % MODIFIED CALL
        end
    end

    Z = reconstruct_tensor(A); % MODIFIED CALL

    % [PSNR_value, SSIM_value, RSE_value] = compute_tensor_metrics(Z, original_tensor);
    
    % PSNR_value = compute_psnr(original_tensor, Z); % MODIFIED CALL
    % SSIM_value = compute_ssim(original_tensor, Z); % MODIFIED CALL
    % RSE_value = compute_rse(original_tensor, Z);   % MODIFIED CALL

    if is_eva
        fprintf("PSNR: %f\n", PSNR_value);
        fprintf("SSIM: %f\n", SSIM_value);
        fprintf("RSE: %f\n", RSE_value);

        varargout{1} = loss_history; 
        varargout{2} = PSNR_history;
        varargout{3} = SSIM_history;
        varargout{4} = RSE_history;
    else
        % fprintf("PSNR: %f, SSIM: %f, RSE: %f, Output Z shape: %s\n", PSNR_value, SSIM_value, RSE_value, mat2str(size(Z)));


        for k_out = 1:4 
            varargout{k_out} = [];
        end
    end
end

