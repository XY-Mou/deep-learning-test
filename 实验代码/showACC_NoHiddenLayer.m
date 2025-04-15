clear all; close all;
load MNISTData X_Test D_Test
load results_NoHiddenLayer.mat

N = length(D_Test);
d_comp = zeros(1,N);

for k = 1:N
    x = reshape(X_Test(:, :, k), 28*28, 1);
    d = D_Test(:, k);

    v = W1 * x;
    y = Softmax(v);
    
    [~, i] = max(y);
    d_comp(k) = i;
end

[~, d_true] = max(D_Test);
correctMsk = (d_comp == d_true);
acc = sum(correctMsk)/N;
fprintf('Accuracy is %f\n', acc);