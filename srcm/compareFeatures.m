function [a,diff_set,frames1] = compareFeatures(im1,im2,imgExtent,decrsThreshold)

    im1 = imresize(im1,0.25);
    im2 = imresize(im2,0.25);

    [frames1, descrs1] = getFeatures(im1, 'peakThreshold', 0.01);
    [frames2, descrs2] = getFeatures(im2, 'peakThreshold', 0.01);

    [nn, dist2] = findNeighbours(descrs1, descrs2, 2);

    matches_2nn = [1:size(nn,2), 1:size(nn,2) ; nn(1,:), nn(2,:)];

    matches_2nn_plano_xy = frames1(1:2,matches_2nn(1,1:(size(matches_2nn,2)/2)));
    matches_2nn_query_xy = frames2(1:2,:);
    matches_all = [1:size(matches_2nn_query_xy,2)];
    matches_near = [];

    for i = 1:size(matches_2nn_plano_xy,2)
        if (matches_2nn_plano_xy(1,i) - imgExtent) >= 0 && (matches_2nn_plano_xy(2,i) - imgExtent) >= 0
            imageStartX = int16(matches_2nn_plano_xy(1,i) - imgExtent);
            imageStartY = int16(matches_2nn_plano_xy(2,i) - imgExtent);
        else
            if (matches_2nn_plano_xy(1,i) - imgExtent) < 0
                imageStartX = 0;
                imageStartY = int16(matches_2nn_plano_xy(2,i) - imgExtent);
            elseif (matches_2nn_plano_xy(2,i) - imgExtent) < 0
                imageStartX = int16(matches_2nn_plano_xy(1,i) - imgExtent);
                imageStartY = 0;
            else 
                imageStartX = 0;
                imageStartY = 0;
            end
        end
        
        for j = 1:size(matches_2nn_query_xy,2)
            if matches_2nn_query_xy(1,j) >= imageStartX &&  matches_2nn_query_xy(1,j) <= (imageStartX + (2 * imgExtent)) && ...
                 (matches_2nn_query_xy(2,j) >= imageStartY) &&  matches_2nn_query_xy(2,j) <= (imageStartY + (2 * imgExtent))
                sumD = sum(descrs1(:,i).*descrs2(:,j))/(norm(descrs1(:,i),2)*norm(descrs2(:,j),2));
                if sumD > decrsThreshold
                    matches_near = [matches_near,[matches_2nn(1,i);matches_all(1,j)]];
                    break;
                end
            end
        end
    end

    matches_2nn_unique = unique(matches_2nn(1,:),'stable');
    matches_near_unique = unique(matches_near(1,:),'stable');
    diff_set = setdiff(matches_2nn_unique,matches_near_unique);



    %figure(1) ;
    %set(gcf,'name', 'correspondence not present in final matches') ;
    %subplot(1,2,1) ; imagesc(im1) ; axis equal off ; hold on ;
    %vl_plotframe(frames1(1:3,diff_set), 'linewidth', 2);

    a = size(diff_set,2)/size(matches_2nn_unique,2);
end