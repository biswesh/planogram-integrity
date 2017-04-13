function matches = computeHog2(im1,im2)
    im1 = imresize(im1,1);
    h1 = localhog(im1);
    im2 = imresize(im2,1);
    h2 = localhog(im2);
    matches = [];

    windowSize = 3;
    yslide = 3;
    xslide = 3;
    diffThreshold = 0.92;



    count = 0;

    shift = (windowSize - 1) / 2;


    for i=(1+(yslide+shift)):(size(h1,1)-(yslide+shift))
        for j = (1+(xslide+shift)):(size(h1,2)-(xslide+shift))
            found = false;
            window1 = [];
            count = count + 1;
            for k = -shift:shift
                for l = -shift:shift
                    val1 = h1(i+k,j+l,:);
                    val1 = val1(:);
                    window1 = [window1;val1];
                 
                end
            end



            for m = (i-yslide):(i+yslide)
                for n = (j-xslide):(j+xslide)
                    window2 = [];
                    for k = -shift:shift
                        for l = -shift:shift
                            val2 = h2(m+k,n+l,:);
                            val2 = val2(:);
                            window2 = [window2;val2];
                        end
                    end


                    if(norm(val1,2) < 1e-8 && norm(val2,2) < 1e-8)
                        found = true;
                        break;
                    end
                    
                    diff = sum(window1.*window2)/((norm(window2,2))*(norm(window2,2)));
                    %diff = (window2-window1).^2;
                    %diff = 2.0*sum(diff./(window2-window1));
                    %diff = diff/numel(window2);
                    if diff > diffThreshold
                        found = true;
                        break;
                    end

                end
                if found
                    break;
                end
            end

            if ~found
                matches = [matches,[i;j]];
            end
        end
    end

    count 

    matches(1,:) = 4+(matches(1,:)-1)*8;
    matches(2,:) = 4+(matches(2,:)-1)*8;
end



