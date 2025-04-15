function y = Conv(x, W)
%
%
%     [wrow, wcol, numFilters] = size(W);
%     [xrow, xcol, ~] = size(x);
%     y = zeros(xrow - wrow + 1, xcol - wcol + 1, numFilters);
%     for k = 1:numFilters
%         y(:, :, k) = conv2(x, rot90(W(:, :, k),2), 'valid');
%     end

    [xrow, xcol, xcha] = size(x);
    [wrow, wcol, wpage, numFilters] = size(W);
    if xcha>1&&(xcha==wpage)
        y = zeros(xrow - wrow + 1, xcol - wcol + 1, numFilters);
        W = W(:,:,end:-1:1,:);
        for i = 1:numFilters
            for j = 1:wpage
                W(:,:,j,i) = rot90(W(:,:,j,i),2);
            end
            y(:, :, i) = convn(x, W(:,:,:,i), 'valid');
        end
    else
        y = zeros(xrow - wrow + 1, xcol - wcol + 1, wpage);
        for k = 1:wpage
            y(:, :, k) = conv2(x, rot90(W(:, :, k),2), 'valid');
        end
    end
    
end

