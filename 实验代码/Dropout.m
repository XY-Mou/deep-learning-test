function ym = Dropout( y, ratio )
%DROPOUT 此处显示有关此函数的摘要
%   此处显示详细说明
    [m, n] = size(y);
    ym = zeros(m,n);
    num = round(m * n * (1 - ratio));
    idx = randperm(m*n, num);
    ym(idx) = m*n / num;
end

