function [W1, W3, W4, W5] = W_update_TwoHiddenLayers(W1, W3, W4, W5, X, D)
%W_UPDATE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
alpha = 0.01;


N = 60000;

for k = 1:N
    x = X(:,:,k);
    d = D(:,k);
   
    dW1 = zeros(9, 9, 20);
    
    %20���˲�����
    V1 = Conv(x, W1);                 % Convolution,  20x20x20
    Y1 = ReLU(V1);                    % 20x20x20
    Y2 = Pool(Y1);
    
    y2 = reshape(Y2, [], 1);
    v3 = W3*y2;
    y3 = ReLU(v3);
    y3 = y3 .* Dropout(y3, 0.2);
    
    v4 = W4*y3;
    y4 = ReLU(v4);
    y4 = y4 .* Dropout(y4, 0.2);
    
    v = W5 * y4;
    y = Softmax(v);
    
    e = d - y;
    delta = e;
    
    e4 = W5' * delta;
    delta4 = (y4>0) .* e4;
    
    e3 = W4' * delta4;
    delta3 = (y3>0) .* e3;
    
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
    
    dW4 = alpha * delta4 * y3';
    W4 = W4 + dW4;
    
    dW5 = alpha * delta * y4';
    W5 = W5 + dW5;
    
end

end

