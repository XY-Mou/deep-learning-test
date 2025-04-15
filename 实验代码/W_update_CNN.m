function [W1, W3, W4] = W_update_CNN(W1, W3, W4, X, D)
%W_UPDATE 此处显示有关此函数的摘要
%   此处显示详细说明
alpha = 0.01;


N = 60000;

for k = 1:N
    x = X(:,:,k);
    d = D(:,k);
    
    dW1 = zeros(9, 9, 20);
    
    %20个滤波器组
    V1 = Conv(x, W1);                 % Convolution,  20x20x20
    Y1 = ReLU(V1);                    % 20x20x20
    Y2 = Pool(Y1);
    
    y2 = reshape(Y2, [], 1);
    v3 = W3*y2;
    y3 = ReLU(v3);
    
    v = W4 * y3;
    y = Softmax(v);
    
    e = d - y;
    delta = e;
    
    e3 = W4' * delta;
    delta3 = (v3>0) .* e3;
    
    e2 = W3' * delta3;
    
    E2 = reshape(e2, size(Y2));
    
    E1 = zeros(size(Y1));
    E2_4= E2/4;
    E1(1:2:end,1:2:end,:) = E2_4;
    E1(1:2:end,2:2:end,:) = E2_4;
    E1(2:2:end,1:2:end,:) = E2_4;
    E1(2:2:end,2:2:end,:) = E2_4;
    
    delta1 = (V1 > 0) .* E1;
    
    for m = 1:20
        dW1(:,:,m) = alpha*filter2(delta1(:,:,m), x, 'valid');
    end
    
    W1 = W1 + dW1;
    
    dW3 = alpha * delta3 * y2';
    W3 = W3 + dW3;
    
    dW4 = alpha * delta * y3';
    W4 = W4 + dW4;
    
end

end

