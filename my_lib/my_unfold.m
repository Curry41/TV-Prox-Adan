function M = my_unfold(T, mode)
    % T: input tensor, can be any N-D
    % mode: mode to unfold (1-based, like MATLAB; 1=first, 2=second, etc.)
    sz = size(T);
    N = ndims(T);

    % Move the desired mode to the front (like np.moveaxis)
    perm = [mode, 1:mode-1, mode+1:N];
    T_perm = permute(T, perm);

    % Reshape to (mode_dim, rest)
    new_shape = [sz(mode), prod(sz)/sz(mode)];
    tmp = reshape(T_perm, new_shape);

    % Now: reorder the columns to match NumPy's row-major
    % We need to generate the correct column ordering
    rest_dims = sz([1:mode-1, mode+1:N]);
    rest_order = 1:numel(tmp)/sz(mode);

    if numel(rest_dims) <= 1
        % 1D case or just 2D: nothing to permute
        idx = rest_order;
    else
        % Generate row-major order (NumPy style) for remaining dims
        idx = reshape(1:prod(rest_dims), rest_dims);
        idx = permute(idx, numel(rest_dims):-1:1); % reverse dims
        idx = idx(:)';
    end
    M = tmp(:, idx);
end