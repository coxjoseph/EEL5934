%% Assignment #6 base code
clc
close all
clear all

%% Object generation


N = 200; % Image size

xc = 100; % Sphere x center
yc = 100; % Sphere y center
r = 25; % Radius of the sphere

a = 10; % Constant signal per pixel

object = zeros(N,N); % Initialization

for i = 1:N
    for j = 1:N
        
        dist = sqrt(((i-xc)^2)+((j-yc)^2));
        flag = (r-dist);
        
        if flag >= 0
            object(i,j) = a;
        else
            object(i,j) = 0;
        end
        
    end
end

figure, imagesc(object), axis image, title('Original image')

% Adding salt and pepper noise 
object_noise = imnoise(object, 'salt & pepper');

% Adding uniform gaussian noise
object_noise = imnoise(object_noise, 'gaussian');
figure, imshow(object_noise), axis image, title('Object + Noise')



