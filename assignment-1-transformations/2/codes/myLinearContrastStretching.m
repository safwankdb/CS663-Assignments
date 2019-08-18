function img = myLinearContrastStretching(org)


org = im2double(org);
% Works for any aspect ratio
org_min = repmat(min(min(org)), size(org, 1), size(org, 2));
org_max = repmat(max(max(org)), size(org, 1), size(org, 2));
img = mat2gray((org - org_min) * 255 ./ (org_max - org_min));

%org = imread('../data/barbara.png');
%img = myLinearContrastStretching(org);

figure('name', 'LinearContrastStretching on Barbara')
colormap(jet(200));
subplot(2, 2, 1), imagesc(org);
title('Barbara')
colorbar;
subplot(2, 2, 2), imagesc(img);
title('After LCS')
colorbar;
subplot(2, 2, 3), imhist(org);
subplot(2, 2, 4), imhist(img);


