clc;
clear;
close all;

video_dir = './TestVideo/';
result_dir = './results_CP_TV_videos/';
img_dir = [result_dir 'imgs/'];
output_csv_detailed = fullfile(result_dir, 'CP_TV_video_metrics_detailed.csv');
output_csv_avg = fullfile(result_dir, 'CP_TV_video_metrics_avg.csv');
miss_rates = [0.8, 0.9, 0.95, 0.97, 0.99];
max_frame = 30;
rank = 200;
num_iteration = 300;
is_add_TV = true;
is_fine_tune = false;
is_eva = false;
is_measure_time = false;
num_trials = 10;

if ~exist(result_dir, 'dir'), mkdir(result_dir); end
if ~exist(img_dir, 'dir'), mkdir(img_dir); end

mat_files = dir(fullfile(video_dir, '*.mat'));

fid1 = fopen(output_csv_detailed, 'w');
fid2 = fopen(output_csv_avg, 'w');
fprintf(fid1, 'VideoName,SR,Trial,PSNR,SSIM,RSE,Time\n');
fprintf(fid2, 'VideoName,SR,MeanPSNR,MeanSSIM,MeanRSE,MeanTime\n');

for fidx = 1:length(mat_files)
    video_path = fullfile(video_dir, mat_files(fidx).name);
    data = load(video_path);
    fn = fieldnames(data);

    % Find first 4D tensor variable
    ref_tensor = [];
    for i = 1:length(fn)
        x = data.(fn{i});
        if isnumeric(x) && ndims(x)==4
            ref_tensor = x;
            break;
        end
    end
    if isempty(ref_tensor)
        warning('No 4D tensor found in %s', video_path);
        continue;
    end
    if max(ref_tensor(:)) > 1
        ref_tensor = double(ref_tensor) / 255.0;
    else
        ref_tensor = double(ref_tensor);
    end

    nf = min(max_frame, size(ref_tensor, 4));
    ref_tensor = ref_tensor(:, :, :, 1:nf);

    tensor_size = size(ref_tensor);
    N = ndims(ref_tensor);
    [~, data_name, ~] = fileparts(mat_files(fidx).name);

    for midx = 1:length(miss_rates)
        miss_rate = miss_rates(midx);
        SR = 1 - miss_rate;

        psnr_list = zeros(1, num_trials);
        ssim_list = zeros(1, num_trials);
        rse_list  = zeros(1, num_trials);
        time_list = zeros(1, num_trials);

        for t = 1:num_trials
            mask_tensor = double(rand(tensor_size) > miss_rate);
            distorted_tensor = mask_tensor .* ref_tensor;

            A_initial = cell(1, N);
            for n = 1:N
                A_initial{n} = rand(tensor_size(n), rank);
            end

            params.lambda_tv = 0.05;
            params.lambda_cp = 0.5;
            params.lambda_3 = 0.1;
         
            params.eta = 0.02;

            params_beta.adan_beta_1 = 0.3;
            params_beta.adan_beta_2 = 0.2;
            params_beta.adan_beta_3 = 0.7;
            params_beta_tv.adan_beta1_tv = 0.3;
            params_beta_tv.adan_beta2_tv = 0.2;
            params_beta_tv.adan_beta3_tv = 0.7;
            beta = [1,1,0,1];

            tStart = tic;
            recon_tensor = TV_Prox_Adan( ...
                A_initial, params, params_beta, params_beta_tv, ...
                distorted_tensor, mask_tensor, beta, num_iteration, ...
                ref_tensor, is_add_TV, is_fine_tune, [], is_eva, is_measure_time);
            time_cost = toc(tStart);

            [psnr_val, rse_val, ssim_val] = computing_metric_order_4(double(recon_tensor)*255, ref_tensor*255);

            psnr_list(t) = psnr_val;
            ssim_list(t) = ssim_val;
            rse_list(t)  = rse_val;
            time_list(t) = time_cost;

            % Save result image for last trial
            if t == num_trials
                frame_idx = unique([1, round(nf/2), nf]);
                for fi = 1:length(frame_idx)
                    frame = frame_idx(fi);
                    img = recon_tensor(:,:,:,frame);
                    if isa(img, 'tensor')
                        img = double(img);
                    end
                    outname = sprintf('%s_CP_TV_SR%.2f_frame%d.png', data_name, SR, frame);
                    imwrite(im2uint8(img), fullfile(img_dir, outname));
                end
            end

            fprintf(fid1, '%s,%.2f,%d,%.6f,%.6f,%.6f,%.4f\n', ...
                data_name, SR, t, psnr_val, ssim_val, rse_val, time_cost);
        end

        % Average metrics
        fprintf(fid2, '%s,%.2f,%.6f,%.6f,%.6f,%.4f\n', data_name, SR, ...
            mean(psnr_list), mean(ssim_list), mean(rse_list), mean(time_list));
    end
end

fclose(fid1);
fclose(fid2);
disp(['Batch completed. Metrics saved to ', output_csv_avg]);
