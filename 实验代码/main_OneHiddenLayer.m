clear all; close all;
load MNISTData

W1 = 2 * rand(100, 28*28) - 1;
W2 = 2 * rand(10, 100) - 1;


for epoch = 1:1
    [W1, W2] = W_update_OneHiddenLayer(W1, W2, X_Train, D_Train);
end


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