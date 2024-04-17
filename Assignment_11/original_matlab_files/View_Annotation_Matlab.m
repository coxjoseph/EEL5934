%% Example reading xml annotation and outputting binary mask


base_dir = 'C:\\Users\\Sam\\Desktop\\Course_Materials_UF\\Post-Doc Assignment\\';
slide_name = 'K1PAS';

% Layer number
structure_idx = 1;

% structure number
image_id = 1;

[bbox_coords, mask_coords] = Read_XML_Annotations(strcat(base_dir,slide_name,'.xml'),structure_idx,image_id);

image = imread(strcat(base_dir,slide_name,'.svs'),'Index',1,'PixelRegion',{bbox_coords(3:4),bbox_coords(1:2)});
mask = poly2mask(mask_coords(:,1),mask_coords(:,2),size(image,1),size(image,2));

% Showing image, mask, and overlay 
figure
subplot(1,3,1),imshow(image),axis image
subplot(1,3,2), imshow(mask), axis image
subplot(1,3,3), imshow(imoverlay(image,mask)), axis image

% Showing overlaid mask with variable transparency
figure
imshow(image), axis image
hold on
h = imagesc(mask);
h.AlphaData = 0.5;
hold off







