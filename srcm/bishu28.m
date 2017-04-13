

clc;
clear;

projectPath = '/home/ubuntu/instanceMatch/iBot/';
codePath = strcat(projectPath,'srcm');
cd(codePath);

run('../vlfeat/toolbox/vl_setup');
addpath(genpath('../piotr_toolbox'));

imgType = '*.jpg';
imgExtent = 25;
decrsThreshold = 0.015;

[imgCells,images] = loadImages(projectPath,imgType);

cd(codePath);

count = 1;
while count < size(imgCells,2) 
    planogramImage = imgCells{1,count};
    planogramImage = imresize(planogramImage,0.5);
    queryImage = imgCells{1,count+1};
    queryImage = imresize(queryImage,0.5);

   


    matches =computeHog2(planogramImage,queryImage);


    [diff_num, diff_set,frames1] = compareFeatures(planogramImage,queryImage,imgExtent,decrsThreshold);
    
    imgName = images(count).name;

    fol = '../testing/results/';
    savePath = strcat(fol,imgName);

    f = figure('visible','off');
    image(planogramImage); hold on; plot(matches(2,:),matches(1,:),'r.','MarkerSize',10); hold on;
    %vl_plotframe(frames1(1:3,diff_set), 'linewidth', 2);
    saveas(f, savePath);
    count = count + 1;
end


    