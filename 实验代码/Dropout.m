function ym = Dropout( y, ratio )
%DROPOUT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    [m, n] = size(y);
    ym = zeros(m,n);
    num = round(m * n * (1 - ratio));
    idx = randperm(m*n, num);
    ym(idx) = m*n / num;
end

