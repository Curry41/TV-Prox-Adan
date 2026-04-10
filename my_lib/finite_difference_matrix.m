function F = finite_difference_matrix(dim_size) % MODIFIED NAME
    if dim_size <= 1
        F = zeros(0, dim_size); 
        return;
    end
    F = zeros(dim_size - 1, dim_size);
    for i = 1:(dim_size - 1)
        F(i, i) = 1;
        F(i, i + 1) = -1;
    end
end