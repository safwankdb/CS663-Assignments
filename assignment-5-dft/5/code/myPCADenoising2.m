function output = myPCADenoising2(img, patch_size, window_size, sigma, k)
    [rows, cols] = size(img);
    dim = 7;
    dim2 = 15;
    K = 200;
    h = waitbar(0,'Applying PCA-2...');
    output = zeros(size(img));
    output_counter = zeros(size(img));
    for i=1:rows-dim+1
        for j=1:cols-dim+1
            ref_patch = img(i:i+dim-1, j:j+dim-1);
            ref_patch = ref_patch(:);
            % First define the neighbourhood centered at i,j
            top_row = max(1, i - dim2);
            down_row = min(cols - dim + 1, i + dim2 - dim + 1);
            left_col = max(1, j - dim2);
            right_col = min(rows - dim + 1, j + dim2 - dim + 1);
            counter = 1;
            patches = zeros([dim * dim, (down_row - top_row + 1) * (right_col - left_col + 1)]);
            mse = zeros([size(patches, 2), 1]);
            for i1=top_row:down_row
                for j1=left_col:right_col
                    patch = img(i1:i1+dim-1, j1:j1+dim-1);
                    patches(:, counter) = patch(:);
                    mse(counter) = norm(ref_patch - patch(:));
                    counter = counter + 1;
                end
            end

            % Extract the `k` nearest neighbours
            [mse_sorted, mse_order] = sort(mse);
            patches = patches(:, mse_order);
            patches = patches(:, 1:min(K, size(patches, 2)));

            num_patches = size(patches, 2);

            p2 = patches * patches';
            [V, D] = eig(p2);

            % Computing the eigen coefficients
            eig_coeff = zeros([dim*dim, num_patches]);
            for i1=1:num_patches
                eig_coeff(:, i1) = V' * patches(:, i1);
            end
            eig_ref = V' * ref_patch;

            % Computing \alpha_j^2
            eig_sq = max(0, mean(eig_coeff .^ 2, 2) - (sigma * sigma));
            % Estimating correct coefficients
            eig_ref_denoised = eig_ref;
            for i1=1:(dim*dim)
                eig_ref_denoised(i1) = eig_ref(i1) / (1 + (sigma * sigma / eig_sq(i1)));
            end

            % Reconstruct original patch and image
            patch_denoised = V * eig_ref_denoised;
            patch_denoised = reshape(patch_denoised, [dim, dim]);
            output(i:i+dim-1, j:j+dim-1) = output(i:i+dim-1, j:j+dim-1) + patch_denoised;
            output_counter(i:i+dim-1, j:j+dim-1) = output_counter(i:i+dim-1, j:j+dim-1) + 1;
        end
        waitbar(i/(rows-dim+1));
    end
    close(h);
    output = output ./ output_counter;
end