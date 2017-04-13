clc;
clear;

run('/home/ubuntu/bixi/koustubh/Documents/vgg_instance_recognition/practical-instance-recognition-2016b/vlfeat/toolbox/vl_setup');

im1 = imread('/home/ubuntu/instanceMatch/iBot/testing/test-data/displaced_2/1.jpg');
%im1 = imresize(im1,0.25);
im2 = imread('/home/ubuntu/instanceMatch/iBot/testing/test-data/displaced_2/1.jpg');
%im2 = imresize(im2,0.25);


[frames1, descrs1] = getFeatures(im1, 'peakThreshold', 0.01);
[frames2, descrs2] = getFeatures(im2, 'peakThreshold', 0.01);

[nn, dist2] = findNeighbours(descrs1, descrs2,2);


matches_2nn = [1:size(nn,2), 1:size(nn,2) ; nn(1,:), nn(2,:)];
[inliers, H] = geometricVerification(frames1, frames2, matches_2nn, 'numRefinementIterations', 4);

matches_geo_refer = matches_2nn(:,inliers);

im3 = imread('/home/ubuntu/instanceMatch/client_data_test/displaced_3/2.jpg');
[frames3, descrs3] = getFeatures(im3, 'peakThreshold', 0.01);
[nn_new, dist2_new] = findNeighbours(descrs1, descrs3,2);
matches_2nn_new = [1:size(nn_new,2), 1:size(nn_new,2) ; nn_new(1,:), nn_new(2,:)];
[inliers_new, H_new] = geometricVerification(frames1, frames3, matches_2nn_new, 'numRefinementIterations', 4);
matches_geo_new = matches_2nn_new(:,inliers_new);



matches_2nn_unique = unique(matches_geo_refer(1,:),'stable');
matches_2nn_new_unique = unique(matches_geo_new(1,:),'stable');
diff_set = setdiff(matches_2nn_unique,matches_2nn_new_unique);

%for i = 1:size(diff_set,2)
%    for j = 1:size(matches_2nn_unique,2)
%        if diff_set(1,i) == matches_2nn_unique(1,j)
%            diff_2nd(1,i) = matches_2nn_unique(2,j);
%        end
%    end
%end

%diff_2nd = uint32(diff_2nd);
%diff_final = [diff_set;diff_2nd];

figure(1);
set(gcf,'name', 'correspondence not present in final matches');
subplot(1,2,1); imagesc(im1); axis equal off; hold on;
vl_plotframe(frames1(1:3,diff_set), 'linewidth', 2);




%Display the matches
figure%(1) ; clf ;
set(gcf,'name', 'Part I.D: SIFT descriptors - geometric verification');
plotMatches(im1,im2,frames1,frames2,matches_geo, 'homography', H);
title('Matches filtered by geometric verification');
