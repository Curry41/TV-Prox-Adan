clear; clc;

% === Configuration ===
img_dir = 'TestImages/';
save_dir = 'results_cp_tv_image/';
if ~exist(save_dir, 'dir'), mkdir(save_dir); end
detailed_csv = fullfile(save_dir, 'cp_tv_detailed_image.csv');
avg_csv = fullfile(save_dir, 'cp_tv_avg_image.csv');

img_types = {'*.bmp','*.png','*.jpg','*.jpeg'};
missing_rates = [0.8, 0.9, 0.95, 0.97, 0.99];
num_trials = 10;
rank = 100;
num_iteration = 300;
beta = [1, 1, 0];

% Flags
is_add_TV = true;
is_fine_tune = false;
is_eva = false;
is_measure_time = false;

% Gather images
img_files = [];
for i = 1:length(img_types)
    img_files = [img_files; dir(fullfile(img_dir, img_types{i}))]; %#ok<AGROW>
end

% Write CSV headers
fid1 = fopen(detailed_csv, 'w');
fid2 = fopen(avg_csv, 'w');
fprintf(fid1, 'Image,MissingRate,Trial,PSNR,SSIM,RSE,TimeCost\n');
fprintf(fid2, 'Image,MissingRate,MeanPSNR,MeanSSIM,MeanRSE,MeanTime\n');

for img_idx = 1:length(img_files)
    img_path = fullfile(img_dir, img_files(img_idx).name);
    [~, base_name, ~] = fileparts(img_files(img_idx).name);
    GT = im2double(imread(img_path)); 
    if size(GT,3) > 3, GT = GT(:,:,1:3); end
    tensor_size = size(GT);
    N = ndims(GT);

    for miss_idx = 1:length(missing_rates)
        miss_rate = missing_rates(miss_idx);

        % Parameter settings for TV only
        params.lambda_tv = 0.01;
        params.lambda_cp = 0.05;
        params.lambda_3 = 0.1;
        params.lambda_4 = 0;
        params.eta = 0.02;
        params_beta.adan_beta_1 = 0.3;
        params_beta.adan_beta_2 = 0.2;
        params_beta.adan_beta_3 = 0.7;
        params_beta_tv.adan_beta1_tv = 0.3;
        params_beta_tv.adan_beta2_tv = 0.2;
        params_beta_tv.adan_beta3_tv = 0.7;
        trans_type = [];

        % Metric accumulators
        psnr_list = zeros(1, num_trials);
        ssim_list = zeros(1, num_trials);
        rse_list  = zeros(1, num_trials);
        time_list = zeros(1, num_trials);

        for trial = 1:num_trials
            mask_tensor = double(rand(tensor_size) > miss_rate);
            distorted_tensor = mask_tensor .* GT;

            A_initial = cell(1, N);
            for n = 1:N
                A_initial{n} = rand(tensor_size(n), rank);
            end

            tStart = tic;
            recon_tensor = TV_Prox_Adan(...
                A_initial, params, params_beta, params_beta_tv, ...
                distorted_tensor, mask_tensor, beta, ...
                num_iteration, GT, is_add_TV, is_fine_tune, trans_type, ...
                is_eva, is_measure_time);
            time_cost = toc(tStart);

            [PSNR_val, SSIM_val, RSE_val] = compute_tensor_metrics(recon_tensor, GT);

            % Save metrics
            psnr_list(trial) = PSNR_val;
            ssim_list(trial) = SSIM_val;
            rse_list(trial)  = RSE_val;
            time_list(trial) = time_cost;

            % Save result image 
            if trial == num_trials
                out_img = recon_tensor;
                if isa(out_img, 'tensor')
                    out_img = double(out_img);
                end
                if size(out_img, 3) == 1
                    out_img = squeeze(out_img);
                end
                outname = sprintf('%s_Miss%.2f_TV.png', base_name, miss_rate);
                imwrite(im2uint8(out_img), fullfile(save_dir, outname));
            end

            fprintf(fid1, '%s,%.2f,%d,%.6f,%.6f,%.6f,%.6f\n', ...
                base_name, miss_rate, trial, ...
                PSNR_val, SSIM_val, RSE_val, time_cost);
        end

        % Write average row
        fprintf(fid2, '%s,%.2f,%.6f,%.6f,%.6f,%.6f\n', ...
            base_name, miss_rate, ...
            mean(psnr_list), mean(ssim_list), mean(rse_list), mean(time_list));
    end
end

fclose(fid1); fclose(fid2);
disp('Batch testing completed.');
