function rgb = yuv2rgbframe(yuv)
    % yuv: [H W 3]
    Y = yuv(:,:,1);
    U = yuv(:,:,2) - 0.5;
    V = yuv(:,:,3) - 0.5;
    R = Y + 1.402 * V;
    G = Y - 0.344136 * U - 0.714136 * V;
    B = Y + 1.772 * U;
    rgb = cat(3, R, G, B);
    rgb = max(0, min(1, rgb)); % clamp
end
