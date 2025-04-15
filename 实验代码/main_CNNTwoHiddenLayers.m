clear all;
close all;
load MNISTData;

% 初始化权值矩阵
W1 = randn(9, 9, 20);
W3 = (2*rand(80,2000)-1)/20;
W4 = (2*rand(100,80)-1)/10;
W5 = (2*rand(10,100)-1)/10;

%更新权值
for epoch = 1:1
    [W1, W3, W4, W5] = W_update_CNNTwoHiddenLayers(W1, W3, W4, W5, X_Train, D_Train);
end

%正确率统计
N = length(D_Test);
d_comp = zeros(1,N);

for k = 1:N
    x = X_Test(:,:,k);
    
    %20个滤波器组
    V1 = Conv(x, W1);                 % Convolution,  20x20x20
    Y1 = ReLU(V1);                    % 20x20x20
    Y2 = Pool(Y1);
    
    y2 = reshape(Y2, [], 1);
    v3 = W3*y2;
    y3 = ReLU(v3);
    v4 = W4* y3;
    y4 = ReLU(v4);
    
    v = W5 * y4;
    y = Softmax(v);
    
    [~, i] = max(y);
    d_comp(k) = i;
end
    
    [~, d_true] = max(D_Test);
    correctMsk = (d_comp == d_true);
    acc = sum(correctMsk)/N;
    fprintf('Accuracy is %f\n', acc);
    
