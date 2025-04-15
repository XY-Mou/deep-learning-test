function [W1, W2, W3] = W_update_TwoHiddenLayers(W1, W2, W3, X, D)
%DEEPRELU_DROPOUT 此处显示有关此函数的摘要
%   此处显示详细说明
      alpha = 0.01;
      N = 60000;  
      for k = 1:N
        x  = reshape(X(:, :, k), [], 1);
        
        v1 = W1 * x;
        y1 = ReLU(v1);
        
        
        v2 = W2 * y1;
        y2 = ReLU(v2);
        
        
        y  = Softmax(W3*y2);

        d = D(:, k);
        
        e = d - y;
        delta = e;
        
        e2     = W3'*delta;
        delta2 = (v2 > 0).*e2;

        e1     = W2'*delta2;
        delta1 = (v1 > 0).*e1;
        
        dW3 = alpha*delta*y2';
        W3  = W3 + dW3;
    
        dW2 = alpha*delta2*y1';
        W2  = W2 + dW2;

        dW1 = alpha*delta1*x';
        W1  = W1 + dW1;
    end
end