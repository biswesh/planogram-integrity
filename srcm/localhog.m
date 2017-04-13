function h = localhog(im)
im = rgb2gray(im);
I=imResample(single(im),[size(im,1) size(im,2)])/255;
im = I;
%im = rgb2gray(im);im = im2single(im);
h = hog(im,8,9,20,1);

