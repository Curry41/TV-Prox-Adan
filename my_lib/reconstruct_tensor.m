% Helper function: reconstruct_tensor
function Z_recon = reconstruct_tensor(A_cell) % MODIFIED NAME
    
    Z_cp = ktensor(A_cell);    % Create a CP tensor (ktensor object)
    Z_recon = full(Z_cp);  % Reconstruct the full tensor
end