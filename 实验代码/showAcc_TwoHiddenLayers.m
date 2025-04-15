clear all; close all;
load MNISTData X_Test D_Test
load results_TwoHiddenLayers.mat

%正确率统计
N = length(D_Test);
d_comp = zeros(1,N);

for k = 1:N
    x = reshape(X_Test(:, :, k), 28*28, 1);
    d = D_Test(:, k);

    v1 = W1 * x;
    y1 = ReLU(v1);
    
    v2 = W2 * y1;
    y2 = ReLU(v2);

    v3 = W3 * y2;
    y3 = Softmax(v3);
    
    [~, i] = max(y3);
    d_comp(k) = i;
end

[~, d_true] = max(D_Test);
correctMsk = (d_comp == d_true);
acc = sum(correctMsk)/N;
fprintf('Accuracy is %f\n', acc);