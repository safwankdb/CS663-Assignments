%% MyMainScript

tic;
%% Your code here

struct_img = load('../data/image_low_frequency_noise.mat');
img = double(struct_img.Z);
Yf = fftshift(fft2(img)); %Fourier Transform & Shift

figure
imshow(img, [], 'InitialMagnification', 'fit');
title('Original Image');
colorbar, axis on;

figure
imshow(log(abs(Yf)+1), [], 'InitialMagnification', 'fit'); 
title('Log Magnitude of Shifted Forurier Transform');
colorbar, axis on; 
impixelinfo; %To find the pixels where the unwanted intensities are appearing

k = 3; %Size of the notch filter = (2k + 1)
notch = -k:k; 
Yf(119 + notch, 124 + notch) = 0; %Dimming the intensities at the notch points
Yf(139 + notch, 134 + notch) = 0;

Y_res = ifft2(ifftshift(Yf)); %Inverse Fourier Transform

figure
imshow(log(abs(Yf)+1), [], 'InitialMagnification', 'fit'); 
title('Log Magnitude after applying Notch Filter of size 7x7');
colorbar, axis on; 

figure
imshow(abs(Y_res), [], 'InitialMagnification', 'fit');
title('Filtered Image');
colorbar, axis on; 

toc;
