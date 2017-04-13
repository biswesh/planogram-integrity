function matches = computeHog(im1,im2)
    im1 = imresize(im1,1);
    h1 = localhog(im1);
    im2 = imresize(im2,1);
    h2 = localhog(im2);
    matches = [];
    %compareBlocks = 5;
    for i=1:size(h1,1)
        for j = 1:size(h1,2)
            found = false;
            for k = -1:1
                for l = -1:1

                    if ((i+k) < 1) || ((j+l) < 1) || ((i+k) > size(h1,1)) || ((j+l) > size(h1,2))
                        continue;
                    end
                    val1 = h1(i,j,:);
                    val2 = h2(i+k,j+l,:);
                    val1 = val1(:);
                    val2 = val2(:);
                    if(norm(val1,2) < 1e-8 && norm(val2,2) < 1e-8)
                        found = true;
                        break;
                    end
                    diff = sum(val1.*val2)/((norm(val1,2))*(norm(val2,2)));
                    if diff > 0.9
                        found = true;
                        break;
                    end
                end
                if found
                    break;
                end
            end
            if ~found
                %disp(norm(val1,2));
                matches = [matches,[i;j]];
            end
        end
    end

    matches(1,:) = 4+(matches(1,:)-1)*8;
    matches(2,:) = 4+(matches(2,:)-1)*8;
end


