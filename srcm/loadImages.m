function [Seq,images] = loadImages(projectPath,imgType)
    imgPath = strcat(projectPath,'testing/test-data/removed_swapped/');
    cd(imgPath);
    images = dir(imgType);
    N = length(images);

    imgPath

    if( ~exist(imgPath, 'dir') || N<1 )
        display('Directory not found or no matching images found.');
    end

    Seq = cell(1,2);
    for idx = 1:2
        Seq{idx} = imread([imgPath images(idx).name]);
    end
end