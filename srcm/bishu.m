
clc;
clear;

run('/home/ubuntu/bixi/koustubh/Documents/vgg_instance_recognition/practical-instance-recognition-2016b/vlfeat/toolbox/vl_setup');

im1 = imread('/home/ubuntu/instanceMatch/client_data_test/first_last/1.jpg');
im1 = imresize(im1,0.25);
im2 = imread('/home/ubuntu/instanceMatch/client_data_test/first_last/2.jpg');
im2 = imresize(im2,0.25);

[frames1, descrs1] = getFeatures(im1, 'peakThreshold', 0.01);
[frames2, descrs2] = getFeatures(im2, 'peakThreshold', 0.01);

%nn = findNeighbours(descrs1, descrs2);
%matches = [1:size(descrs1,2) ; nn(1,:)];



%Find the top two neighbours as well as their distances
[nn, dist2] = findNeighbours(descrs1, descrs2, 2) ;

%idx = dist2(1,:) < 0.2;    %new addition
%dist2 = dist2(:,idx);       %new addition
%nn = nn(:,idx);           %new addition

% Accept neighbours if their second best match is sufficiently far off
%nnThreshold = 0.8;
%ratio2 = dist2(1,:) ./ dist2(2,:) ;
%ok = ratio2 <= nnThreshold^2 ;

% Construct a list of filtered matches
%matches_2nn = [find(ok) ; nn(1, ok)] ;

matches_2nn = [1:size(nn,2), 1:size(nn,2) ; nn(1,:), nn(2,:)];


image_extent = 25;
decrs_threshold = 0.8;



matches_2nn_plano_xy = frames1(1:2,matches_2nn(1,1:(size(matches_2nn,2)/2)));
matches_2nn_query_xy = frames2(1:2,:);
matches_all = [1:size(matches_2nn_query_xy,2)];
matches_near = [];



for i = 1:size(matches_2nn_plano_xy,2)
    if (matches_2nn_plano_xy(1,i) - image_extent) >= 0 && (matches_2nn_plano_xy(2,i) - image_extent) >= 0
        imageStartX = int16(matches_2nn_plano_xy(1,i) - image_extent);
        imageStartY = int16(matches_2nn_plano_xy(2,i) - image_extent);
    else
        if (matches_2nn_plano_xy(1,i) - image_extent) < 0
            imageStartX = 0;
            imageStartY = int16(matches_2nn_plano_xy(2,i) - image_extent);
        elseif (matches_2nn_plano_xy(2,i) - image_extent) < 0
            imageStartX = int16(matches_2nn_plano_xy(1,i) - image_extent);
            imageStartY = 0;
        else 
            imageStartX = 0;
            imageStartY = 0;
        end
    end
        
    for j = 1:size(matches_2nn_query_xy,2)
        if matches_2nn_query_xy(1,j) >= imageStartX &&  matches_2nn_query_xy(1,j) <= (imageStartX + (2 * image_extent)) && ...
             (matches_2nn_query_xy(2,j) >= imageStartY) &&  matches_2nn_query_xy(2,j) <= (imageStartY + (2 * image_extent))
            sumD = sum(descrs1(:,i).*descrs2(:,j))/(norm(descrs1(:,i),2)*norm(descrs2(:,j),2));
            if sumD > decrs_threshold
                matches_near = [matches_near,[matches_2nn(1,i);matches_all(1,j)]];
                break;
            end
        end
    end
end



%frames_new = frames1(:,idx);    %new addition
%frames_new=frames1;             %new addition                                 


%[inliers, H] = geometricVerification(frames1, frames2, matches_2nn, 'numRefinementIterations', 4) ;

%{
numMatches = size(matches_2nn,2);
inliers = cell(1, numMatches);
H = cell(1, numMatches);

%frames_new = frames1(:,idx);         %new addition
%frames_new=frames1;                  %new addition
x1 = double(frames1(1:2, matches_2nn(1,:))) ;   
x2 = double(frames2(1:2, matches_2nn(2,:))) ;
x1hom = x1;
x2hom = x2;
x1hom(end+1,:) = 1;
x2hom(end+1,:) = 1;

tolerance1 = 20;
tolerance2 = 15;
minInliers = 6;

for m = 1:numMatches
    for t = 1:4
    	if t == 1
    		A1 = toAffinity(frames1(:,matches_2nn(1,m)));   %new addition
    	    A2 = toAffinity(frames2(:,matches_2nn(2,m)));
    	    H21 = A2 * inv(A1);
    	    x1p = H21(1:2,:) * x1hom;
    	    tol = tolerance1;
    	else t <= 4
        	H21 = x2(:,inliers{m}) / x1hom(:,inliers{m}) ;
        	x1p = H21(1:2,:) * x1hom ;
        	H21(3,:) = [0 0 1];
        	tol = tolerance2;
        end
        dist3 = sum((x2 - x1p).^2,1);
        inliers{m} = find(dist3 < tol^2);
        H{m} = H21;
        if numel(inliers{m}) < minInliers, break ; end
        if numel(inliers{m}) > 0.7 * size(matches_2nn,2), break ; end % enough!
    end
end
scores = cellfun(@numel, inliers);
[~, best] = max(scores); 
inliers = inliers{best};
H = inv(H{best});
%}


%matches_geo = matches_2nn(:,inliers);

matches_2nn_unique = unique(matches_2nn(1,:),'stable');
%matches_geo_unique = unique(matches_geo(1,:),'stable');



diff_set = setdiff(matches_2nn_unique,matches_near(1,:));

size(diff_set,2)
%diff_2nd = zeros(1,size(diff_set,2));

%for i = 1:size(diff_set,2)
%    for j = 1:
%        if diff_set(1,i) == matches_2nn(1,j)
%            diff_2nd(1,i) = matches_2nn(2,j);
%        end
%    end
%end


%Display the images
figure(1) ;
set(gcf,'name', 'correspondence not present in final matches') ;
subplot(1,2,1) ; imagesc(im1) ; axis equal off ; hold on ;
vl_plotframe(frames1(1:3,diff_set), 'linewidth', 2);

%diff_2nd = uint32(diff_2nd);
%diff_final = [diff_set;diff_2nd];

%ximg = frames1(1:2,diff_final(1,:));
%xproj = frames1(1:2,diff_final(2,:));


%dist3 = sum((ximg - xproj).^2,1) ;
%idx = dist3(1,:) < 4000;
%diff_final_mod = diff_final(:,idx);



%Display the images
%figure(1) ;
%set(gcf,'name', 'correspondence not present in final matches') ;
%subplot(1,2,1) ; imagesc(im1) ; axis equal off ; hold on ;
%vl_plotframe(frames1(1:3,diff_final(1,:)), 'linewidth', 2);
%subplot(1,2,2) ; imagesc(im2) ; axis equal off ; hold on;   
%vl_plotframe(frames2(1:3,dif(2,:)), 'linewidth', 2);



%Display the matches
%figure%(1) ; clf ;
%set(gcf,'name', 'Part I.D: SIFT descriptors - geometric verification');
%plotMatches(im1,im2,frames1,frames2,matches_near, 'homography', H);
%title('Matches filtered by geometric verification');




