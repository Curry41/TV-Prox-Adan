% test_performance.m
% Test script to load image, generate a mask, and initialize factor matrices

% --- Parameters ---
filepath = 'TestImages/house.bmp';   % Set image path
miss_rate = 0.99;            % Missing rate: fraction of missing entries
rank = 100;                  % CP decomposition rank
num_iteration = 300;
is_add_TV = true;       % Enable Total Variation regularization
is_fine_tune = false;   % Fine-tuning flag

trans_type = [];     

is_eva = false;          % Evaluate metrics during iterations
is_measure_time = false; % Measure time for parts of the iteration

% --- Step 1: Read and normalize image ---
img = im2double(imread(filepath));  % Convert to [0,1]
ref_tensor = img;
tensor_size = size(ref_tensor);
N = ndims(ref_tensor);

% --- Step 2: Generate binary mask tensor ---
mask_tensor = double(rand(tensor_size) > miss_rate);  % 1: known, 0: missing

distorted_tensor = mask_tensor .* ref_tensor;

% --- Step 3: Initialize CP factor matrices ---
A_initial = cell(1, N);
for n = 1:N
    A_initial{n} = rand(tensor_size(n), rank);
end

% Main parameters
params.lambda_tv = 0.01;   % Weight for TV regularization
params.lambda_cp = 0.05;      % Weight for cp term
params.lambda_3 = 0.1;  
params.eta = 0.02;       % Learning rate


% Adan optimizer parameters for CP part
params_beta.adan_beta_1 = 0.3;
params_beta.adan_beta_2 = 0.2;
params_beta.adan_beta_3 = 0.7;

% Adan optimizer parameters for TV part
params_beta_tv.adan_beta1_tv = 0.3;
params_beta_tv.adan_beta2_tv = 0.2;
params_beta_tv.adan_beta3_tv = 0.7;
fprintf('Parameters defined.\n');

beta = [1,1,0];

tStart = tic;

[recon_tensor] = TV_Prox_Adan(A_initial, params, params_beta, params_beta_tv, distorted_tensor, mask_tensor, beta, num_iteration, ref_tensor, is_add_TV, is_fine_tune, trans_type, is_eva, is_measure_time);



time_cost_I = toc(tStart);

[PSNR_value, SSIM_value, RSE_value] = compute_tensor_metrics(recon_tensor, ref_tensor);

 fprintf("PSNR: %f\n", PSNR_value);
 fprintf("SSIM: %f\n", SSIM_value);
 fprintf("RSE: %f\n", RSE_value);
 fprintf("time: %f\n", time_cost_I);