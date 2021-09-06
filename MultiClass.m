function [W1, W2] = MultiClass(W1, W2, X, D)
%
% W1: input layer - hidden layer Weight function
% W2: hidden layer - output layer Weight function
% X: input data from Leaning data
% D: answer of Learning data
%
alpha = 0.9;

N = 5;

for k = 1:N
    x = reshape(X(:,:,k),25,1); % k번째 이미지 데이터(5x5)를 (25x1)행렬로 reshape 
    d = D(k,:)'; % true,
    
    v1 = W1 * x;
    y1 = Sigmoid(v1);
    
    v = W2 * y1;
    y = Softmax(v);
    
    e = d - y;
    delta = e;
    
    e1 = W2'*delta;
    delta1 = y1.*(1-y1).*e1;
    
    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;
    
    dW2 = alpha*delta*y1';
    W2 = W2 + dW2;
end
end

    