function [W1, W2] = W_update_OneHiddenLayer(W1, W2, X, D)
%DEEPRELU_DROPOUT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
      alpha = 0.01;
      N = 60000;  
      for k = 1:N
        x  = reshape(X(:, :, k), [], 1);
        
        v1 = W1 * x;
        y1 = ReLU(v1);

        y  = Softmax(W2*y1);

        d = D(:, k);
        
        e = d - y;
        delta = e;

        e1     = W2'*delta;
        delta1 = (v1 > 0).*e1;
    
        dW2 = alpha*delta*y1';
        W2  = W2 + dW2;

        dW1 = alpha*delta1*x';
        W1  = W1 + dW1;
    end
end