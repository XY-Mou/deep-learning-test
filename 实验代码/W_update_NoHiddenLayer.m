function [W1] = W_update_NoHiddenLayer(W1, X, D)
%SGD �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
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

