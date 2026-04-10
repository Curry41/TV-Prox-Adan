function X = load_MSI(dir_path)
% Read ALL .png files in dir_path into [H x W x B], single in [0,1].
% Bands are sorted by the first integer found in each filename.
% If no integer is found, filenames are sorted lexicographically.

    if ~isfolder(dir_path)
        error('Folder not found: %s (pwd: %s)', dir_path, pwd);
    end

    L = dir(fullfile(dir_path, '*.png'));     % case-sensitive on *nix; see note below
    if isempty(L)
        % Try uppercase too (Windows/MATLAB usually matches both, but just in case)
        L = dir(fullfile(dir_path, '*.PNG'));
    end
    if isempty(L)
        error('No PNG files in %s. Try: dir(fullfile(dir_path,''*.png''))', dir_path);
    end

    names = {L.name}';
    nums = nan(numel(names),1);
    for i = 1:numel(names)
        t = regexp(names{i}, '\d+', 'match', 'once');   % first integer in name
        if ~isempty(t), nums(i) = str2double(t); end
    end

    % sort: numeric if available, else lexicographic
    if any(~isnan(nums))
        [~, idx] = sort(nums);
    else
        [~, idx] = sort(lower(names));
    end
    names = names(idx);

    % read first to size; collapse RGB to single channel if present
    I0 = imread(fullfile(dir_path, names{1}));
    if ndims(I0) == 3
        % If each PNG is a single band stored as RGB, collapse to gray.
        I0 = rgb2gray(I0);
    end
    [H,W] = size(I0);
    B = numel(names);
    X = zeros(H,W,B,'double');
    X(:,:,1) = im2double(I0);

    for k = 2:B
        Ik = imread(fullfile(dir_path, names{k}));
        if ndims(Ik) == 3, Ik = rgb2gray(Ik); end
        if ~isequal(size(Ik), [H W])
            error('Size mismatch at %s: got %s, expected %dx%d', ...
                  names{k}, mat2str(size(Ik)), H, W);
        end
        X(:,:,k) = im2double(Ik);
    end
end
