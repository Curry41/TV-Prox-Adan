function tensor = yuv2tensor(filename, width, height, nframes)
    tensor = zeros(height, width, 3, nframes, 'double');
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    y_size = width * height;
    uv_width = width / 2;
    uv_height = height / 2;
    uv_size = uv_width * uv_height;

    for f = 1:nframes
        Y = fread(fid, y_size, 'uint8');
        if numel(Y) < y_size, break; end
        Y = reshape(Y, width, height)';

        U = fread(fid, uv_size, 'uint8');
        if numel(U) < uv_size, break; end
        U = reshape(U, uv_width, uv_height)';

        V = fread(fid, uv_size, 'uint8');
        if numel(V) < uv_size, break; end
        V = reshape(V, uv_width, uv_height)';

        U_upsampled = imresize(U, [height width], 'nearest');
        V_upsampled = imresize(V, [height width], 'nearest');

        tensor(:,:,1,f) = double(Y) / 255;            % Y
        tensor(:,:,2,f) = double(U_upsampled) / 255;  % U
        tensor(:,:,3,f) = double(V_upsampled) / 255;  % V
    end

    fclose(fid);
end
