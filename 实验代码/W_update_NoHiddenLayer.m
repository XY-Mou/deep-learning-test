function [W1] = W_update_NoHiddenLayer(W1, X, D)
%SGD 此处显示有关此函数的摘要
%   此处显示详细说明
    alpha = 0.01;
    N = 60000;
    for k = 1:N
    x = reshape(X(:, :, k), 28*28, 1);
    d = D(:, k);

    v = W1 * x;
    y = Softmax(v);
    
    e = d - y;
    delta = e;
    
    dW1 = alpha * delta * x';
    W1 = W1 + dW1;
    end
end

