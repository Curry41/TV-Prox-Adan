function A_kr = khatrirao_except_n(A, n)
    % A is a cell array of factor matrices
    N = numel(A);
    idx = [1:n-1, n+1:N]; % All indices except n
    A_others = A(idx);    % Select all but A{n}
    A_kr = khatrirao(A_others); % Use Tensor Toolbox's function
end