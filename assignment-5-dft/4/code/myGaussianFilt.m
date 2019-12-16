function[] = myGaussianFilt(sigma, img)


[r, c] = size(img);
img_padded = padarray(img, [r/2 c/2]);
[rp, cp] = size(img_padded);

Y = fftshift(fft2(img_padded));
myFilt = zeros(size(Y));

figure
subplot(2, 2, 1), imshow(log(abs(Y) + 1), [], 'InitialMagnification', 'fit');
title('Log Magnitude of F.T. of original image');
colorbar, axis on;


for i = 1:rp
    for j = 1:cp
        myFilt(i, j) = exp(-((i-r)^2+(j-c)^2)/(2*sigma^2))/sqrt(2*pi*sigma^2); %Gaussian Filter
    end
end

myFilt = myFilt/max(max(myFilt));

Yn = ifft2(ifftshift(Y.*myFilt));
Yn_unpadded = Yn((r/2)+1:r+(r/2), (c/2)+1:c+(c/2));


subplot(2, 2, 2), imshow(myFilt, [], 'InitialMagnification', 'fit');
title(strcat('Gaussian Low Pass Filter with sigma = ', num2str(sigma)));
colorbar, axis on;

subplot(2, 2, 3), imshow(log(abs(Y.*myFilt) + 1), [], 'InitialMagnification', 'fit');
title('Log Magnitude of F.T. of filtered image');
colorbar, axis on;

subplot(2, 2, 4), imshow(uint8(abs(Yn_unpadded)), [], 'InitialMagnification', 'fit');
title('Filtered Image');
colorbar, axis on;
%Plotting the original and filtered image
figure
subplot(1, 2, 1), imshow(uint8(img), [], 'InitialMagnification', 'fit');
title('Original Image');
colorbar, axis on;

subplot(1, 2, 2), imshow(uint8(abs(Yn_unpadded)), [], 'InitialMagnification', 'fit');
title('Filtered Image');
colorbar, axis on;
end