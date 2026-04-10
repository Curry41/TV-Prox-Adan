% test_performance_msi.m
% Load MSI cube from a directory of per-band images, generate mask, run CP-TV.

%% --- User params ---
msi_dir     = './MSI/pompoms_ms';       % <--- folder containing band images
pattern_str = 'pompoms_ms_(\d+)\.png'; % numeric band index in filename
miss_rate   = 0.99;                 % fraction missing
rank        = 250;
num_iteration = 300;

is_add_TV      = true;
is_fine_tune   = false;
trans_type     = [];               % 'dct' | 'fft' | []
is_eva         = false;
is_measure_time= false;
is_rse_time    = false;

% Main params
params.lambda_tv = 0.01;    % TV weight
params.lambda_cp = 0.05;    % CP data term weight
params.lambda_3 = 0.10;    % L2 weight decay
params.eta      = 0.02;    % LR


% Adan (CP)
params_beta.adan_beta_1 = 0.3;
params_beta.adan_beta_2 = 0.2;
params_beta.adan_beta_3 = 0.7;
% Adan (TV)
params_beta_tv.adan_beta1_tv = 0.3;
params_beta_tv.adan_beta2_tv = 0.2;
params_beta_tv.adan_beta3_tv = 0.7;

% TV along [row, col, spectral]; set last 0 if you want spatial-only TV.
beta = [1, 1, 1];



fprintf('Loading MSI cube from "%s"...\n', msi_dir);
ref_tensor = load_MSI(msi_dir);   % H x W x B, single in [0,1]
tensor_size = size(ref_tensor);
N = ndims(ref_tensor);

%% --- Mask & distorted tensor ---
mask_tensor = double(rand(tensor_size) > miss_rate);    % 1 known, 0 missing
distorted_tensor = mask_tensor .* ref_tensor;

%% --- Init CP factors ---
A_initial = cell(1, N);
for n = 1:N
    A_initial{n} = rand(tensor_size(n), rank);
end

fprintf('Parameters defined. Running optimizer...\n');
tStart = tic;
[recon_tensor] = TV_Prox_Adan( ...
    A_initial, params, params_beta, params_beta_tv, ...
    distorted_tensor, mask_tensor, beta, num_iteration, ...
    ref_tensor, is_add_TV, is_fine_tune, trans_type, ...
    is_eva, is_measure_time);
time_cost_I = toc(tStart);


[PSNR_value, RSE_value, SSIM_value] = computing_metric_order_3(double(recon_tensor)*255, ref_tensor*255);
fprintf("PSNR: %f\n", PSNR_value);
fprintf("SSIM: %f\n", SSIM_value);
fprintf("RSE: %f\n", RSE_value);
fprintf("time: %f s\n", time_cost_I);

