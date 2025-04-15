clear all; close all;
load MNISTData X_Test D_Test
load results_CNNTwoHiddenLayers.mat

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