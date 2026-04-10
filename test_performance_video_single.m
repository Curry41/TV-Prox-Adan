% test_performance_video.m

% --- Parameters ---
video_path = 'TestVideo/suzie_video_rgb.mat';   % Set video path
var_name = 'video_tensor';               % Name of variable in .mat file (change as needed)
miss_rate = 0.99;            % Missing rate: fraction of missing entries
rank = 200;                 % CP decomposition rank
num_iteration = 300;
is_add_TV = true;           % Enable Total Variation regularization
is_fine_tune = false;       % Fine-tuning flag
max_frame = 30;             % <<<<<<<<<<<< Set maximum number of frames here

trans_type = [];           
is_eva = false;             % Evaluate metrics during iterations
is_measure_time = false;    % Measure time for parts of the iteration

% --- Step 1: Load and normalize video tensor ---
data = load(video_path);
if isfield(data, var_name)
    ref_tensor = data.(var_name);
else
    % if var_name is not known, pick first 4D numeric variable
    fn = fieldnames(data);
    ref_tensor = [];
    for i = 1:length(fn)
        x = data.(fn{i});
        if isnumeric(x) && ndims(x)==4
            ref_tensor = x;
            break;
        end
    end
    if isempty(ref_tensor)
        error('No 4D tensor found in %s', video_path);
    end
end
if max(ref_tensor(:)) > 1
    ref_tensor = double(ref_tensor) / 255.0;
else
    ref_tensor = double(ref_tensor);
end

% --- Only use the first max_frame frames (if there are that many) ---
nf = min(max_frame, size(ref_tensor, 4));
ref_tensor = ref_tensor(:, :, :, 1:nf);

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
params.lambda_tv = 0.05;   % TV regularization weight
params.lambda_cp = 0.5;    % CP term weight
params.lambda_3 = 0.1;    % Proximal term weight
params.eta = 0.02;        % Learning rate

% Adan optimizer parameters for CP part
params_beta.adan_beta_1 = 0.3;
params_beta.adan_beta_2 = 0.2;
params_beta.adan_beta_3 = 0.7;

% Adan optimizer parameters for TV part
params_beta_tv.adan_beta1_tv = 0.3;
params_beta_tv.adan_beta2_tv = 0.2;
params_beta_tv.adan_beta3_tv = 0.7;
fprintf('Parameters defined.\n');

beta = [1,1,0,1]; % Adjust as needed for your TV/transform regularization

tStart = tic;

[recon_tensor] = TV_Prox_Adan( ...
    A_initial, params, params_beta, params_beta_tv, distorted_tensor, mask_tensor, beta, ...
    num_iteration, ref_tensor, is_add_TV, is_fine_tune, trans_type, is_eva, is_measure_time);

time_cost_I = toc(tStart);

[PSNR_value, RSE_value, SSIM_value] = computing_metric_order_4(double(recon_tensor)*255, ref_tensor*255);

 fprintf("PSNR: %f\n", PSNR_value);
 fprintf("SSIM: %f\n", SSIM_value);
 fprintf("RSE: %f\n", RSE_value);
 fprintf("time: %f\n", time_cost_I);


