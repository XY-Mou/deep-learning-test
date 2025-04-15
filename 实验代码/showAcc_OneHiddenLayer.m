clear all; close all;
load MNISTData X_Test D_Test
load results_OneHiddenLayer.mat

N = length(D_Test);
d_comp = zeros(1,N);

for k = 1:N
    x = reshape(X_Test(:, :, k), 28*28, 1);
    d = D_Test(:, k);

    v1 = W1 * x;
    y1 = ReLU(v1);

    v2 = W2 * y1;
    y2 = Softmax(v2);
    
    [~, i] = max(y2);
    d_comp(k) = i;
end

[~, d_true] = max(D_Test);
correctMsk = (d_comp == d_true);
acc = sum(correctMsk)/N;
fprintf('Accuracy is %f\n', acc);