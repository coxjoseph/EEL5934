%% 2-D Convolution example
clc
close all
clear all

% Random input image
x = round(10.*rand(5));
    
% Random convolution kernel
h = rand(3)>0.5;

% Visualizing image and kernel
figure, subplot(1,2,1), imagesc(x), axis image, title('Random Input Image')
subplot(1,2,2), imshow(h), axis image, title('Random Convolution Kernel')

% Convolving two functions
x_conv_h = conv2(x,h,'full');

% Visualizing convolved image
figure, imagesc(x_conv_h), axis image, title('Image Convolved with PSF')




