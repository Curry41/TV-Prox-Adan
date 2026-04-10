function [avg_psnr, avg_ssim, rse] = compute_tensor_metrics(recon_tensor, ref_tensor)
% Compute PSNR, SSIM, and RSE between two tensors in [0,1]
% Supports 3D (image) and 4D (video) tensors
% - recon_tensor: reconstructed tensor (HxWxC or HxWxCxT)
% - ref_tensor: reference tensor (same size)
% - Returns average PSNR, SSIM, and RSE

    assert(all(size(recon_tensor) == size(ref_tensor)), 'Input tensors must be the same size.');
    % assert(isfloat(recon_tensor) && isfloat(ref_tensor), 'Tensors must be float.');
    % assert(all(recon_tensor(:) >= 0 & recon_tensor(:) <= 1), 'Tensors must be in [0,1].');

    tensor_size = size(ref_tensor);
    num_dims = ndims(ref_tensor);

    recon_tensor = double(recon_tensor);

    total_psnr = 0;
    total_ssim = 0;
    num_frames = 1;

    if num_dims == 3
        % 3D tensor: process as one image
        total_psnr = psnr(recon_tensor, ref_tensor);
        total_ssim = ssim(recon_tensor, ref_tensor);
    elseif num_dims == 4
        % 4D tensor: process frame-by-frame
        num_frames = tensor_size(4);
        for t = 1:num_frames
            frame_ref = ref_tensor(:,:,:,t);
            frame_recon = recon_tensor(:,:,:,t);
            total_psnr = total_psnr + psnr(frame_recon, frame_ref);
            total_ssim = total_ssim + ssim(frame_recon, frame_ref);
        end
    else
        error('Only 3D and 4D tensors are supported.');
    end

    % Average over frames if needed
    avg_psnr = total_psnr / num_frames;
    avg_ssim = total_ssim / num_frames;

    % Relative Squared Error (RSE)
    rse = norm(ref_tensor(:) - recon_tensor(:), 2) / norm(ref_tensor(:), 2);
end
