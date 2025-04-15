function y = Pool(x)
%     
% 2x2 mean pooling
%
%
y=(x(1:2:end,1:2:end,:)+x(2:2:end,1:2:end,:)+x(1:2:end,2:2:end,:)+x(2:2:end,2:2:end,:))/4;
end
 