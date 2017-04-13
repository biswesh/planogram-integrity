addpath(genpath('/home/ubuntu/instanceMatch/piotr_toolbox'));
im1 = imread('../test-data/jhadu/1.jpg');
im1 = imresize(im1,0.5);
h1 = localhog(im1,0.5);

im2 = imread('../test-data/jhadu/2.jpg');
im2 = imresize(im2,[1064*2 800*2]);

h2 = localhog(im2);
matches = [];
for i=1:size(h1,1)
    for j = 1:size(h1,2)
        val1 = h1(i,j,:);
        val2 = h2(i,j,:);
        val1 = val1(:);
        val2 = val2(:);
        
        diff = sum(val1.*val2)/((norm(val1,2)+1e-5)*(norm(val2,2)+1e-5));
        %&& norm(val1,2) > 1e-5 
        if diff < 0.8 && norm(val1,2) > 1e-4
            matches = [matches,[i;j]];
        end
    end
end

matches(1,:) = 4+(matches(1,:)-1)*8;
matches(2,:) = 4+(matches(2,:)-1)*8;

image(im1); hold on; plot(matches(2,:),matches(1,:),'r.','MarkerSize',20); 