function img = myHE(org)

mycdf = zeros(256, size(org, 3)); % Treating channels independently
for i = 1:size(org, 3)
    mycdf(:, i) = cumsum(imhist(org(:,:,i))) / numel(org(:,:,1));
end

img = zeros(size(org)); % Pre-allocating
for k = 1:size(org, 3)
    % Computing independently for each channel
    cdfi = mycdf(:,k);
    img(:,:,k) = cdfi(org(:,:,k)+1);
end

figure('name', 'Global HE on Church');
colormap(jet(200));
subplot(2, 2, 1), imagesc(org);
colorbar;
subplot(2, 2, 2), imagesc(img);
colorbar;
subplot(2, 2, 3), imhist(org);
subplot(2, 2, 4), imhist(img);