clear; clc;

% === Configuration ===
msi_root_dir = 'MSI/';  % Parent folder containing multiple MSI subfolders
save_dir = 'results_cp_tv_avg_msi_t/';
if ~exist(save_dir, 'dir'), mkdir(save_dir); end
detailed_csv = fullfile(save_dir, 'cp_tv_detailed.csv');
avg_csv = fullfile(save_dir, 'cp_tv_avg.csv');

missing_rates = [0.8, 0.9, 0.95, 0.97, 0.99];
num_trials = 10;
rank = 250;
num_iteration = 300;
beta = [1, 1, 1];

% Flags
is_add_TV = true;
is_fine_tune = false;
is_eva = false;
is_measure_time = false;

% Write CSV headers
fid1 = fopen(detailed_csv, 'w');
fid2 = fopen(avg_csv, 'w');
fprintf(fid1, 'MSI,SR,Trial,PSNR,SSIM,RSE,TimeCost\n');
fprintf(fid2, 'MSI,SR,MeanPSNR,MeanSSIM,MeanRSE,MeanTime\n');

% === Find all MSI folders ===
msi_folders = dir(msi_root_dir);
msi_folders = msi_folders([msi_folders.isdir] & ~ismember({msi_folders.name},{'.','..'}));

for msi_idx = 1:length(msi_folders)
    msi_name = msi_folders(msi_idx).name;
    msi_dir = fullfile(msi_root_dir, msi_name);

    fprintf('Processing MSI: %s\n', msi_name);

    % Load MSI cube
    ref_tensor = double(load_MSI(msi_dir)); % H x W x B, in [0,1]
    tensor_size = size(ref_tensor);
    N = ndims(ref_tensor);

    for miss_idx = 1:length(missing_rates)
        miss_rate = missing_rates(miss_idx);
        SR = 1 - miss_rate;  % sampling ratio

        % Parameter settings
        params.lambda_tv = 0.01;
        params.lambda_cp = 0.05;
        params.lambda_3 = 0.1;
     
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
            distorted_tensor = mask_tensor .* ref_tensor;

            A_initial = cell(1, N);
            for n = 1:N
                A_initial{n} = rand(tensor_size(n), rank);
            end

            tStart = tic;
            recon_tensor = TV_Prox_Adan(...
                A_initial, params, params_beta, params_beta_tv, ...
                distorted_tensor, mask_tensor, beta, ...
                num_iteration, ref_tensor, is_add_TV, is_fine_tune, trans_type, ...
                is_eva, is_measure_time);
            time_cost = toc(tStart);

            % Compute metrics
            [PSNR_val, RSE_val, SSIM_val] = computing_metric_order_3(double(recon_tensor)*255, ref_tensor*255);

            % Save metrics
            psnr_list(trial) = PSNR_val;
            ssim_list(trial) = SSIM_val;
            rse_list(trial)  = RSE_val;
            time_list(trial) = time_cost;

            % Save all bands of last trial
            if trial == num_trials
                save_band_dir = fullfile(save_dir, sprintf('%s_SR%.2f', msi_name, SR));
                if ~exist(save_band_dir, 'dir'), mkdir(save_band_dir); end
                recon_tensor = double(recon_tensor);
                for b = 1:size(recon_tensor, 3)
                    band_img = recon_tensor(:,:,b);
                    outname = sprintf('Band%02d.png', b);
                    imwrite(im2uint8(band_img), fullfile(save_band_dir, outname));
                end
            end

            fprintf(fid1, '%s,%.2f,%d,%.6f,%.6f,%.6f,%.6f\n', ...
                msi_name, SR, trial, PSNR_val, SSIM_val, RSE_val, time_cost);
        end

        % Write average row
        fprintf(fid2, '%s,%.2f,%.6f,%.6f,%.6f,%.6f\n', ...
            msi_name, SR, mean(psnr_list), mean(ssim_list), mean(rse_list), mean(time_list));
    end
end

fclose(fid1); fclose(fid2);
disp('Batch MSI testing completed.');
