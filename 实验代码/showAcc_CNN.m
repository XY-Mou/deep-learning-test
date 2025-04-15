clear all; close all;
load MNISTData X_Test D_Test
load results_CNN.mat

N = length(D_Test);
d_comp = zeros(1,N);

for k = 1:N
  X = X_Test(:, :, k);              % Input,           28x28
  
  V1 = Conv(X, W1);                 % Convolution,  20x20x20
  Y1 = ReLU(V1);                    % 20x20x20
  Y2 = Pool(Y1);                    % Pool, 10x10x20
  
  y2 = reshape(Y2, [], 1);          % 2000x1  
  v3 = W3*y2;                       %  100x1
  y3 = ReLU(v3);                    % 100x1
  v  = W4*y3;                       % 10x1
  y  = Softmax(v);                  %10x1

  [~, i] = max(y);
  d_comp(k) = i;
end
[~, d_true] = max(D_Test);
correctMsk = (d_comp == d_true);
acc = sum(correctMsk)/N;
fprintf('Accuracy is %f\n', acc);