function [W1, W2] =  BackpropXOR(W1, W2, X, D) 

alpha = 0.9;

N = 4;
for k = 1:N
    
    % select k'th data
    x = X(k,:)';
    d = D(k);
    
    % weighted sum of input nodes
    v1 = W1 * x; 
    y1 = Sigmoid(v1); % activation func.
    
    % weighted sum of hidden nodes
    v = W2 * y1;
    y = Sigmoid(v); % activation func.
    
    % error
    e = d - y;
    delta = y.*(1-y).*e;
     
    % delta of output nodes
    e1 = W2' * delta;
    delta1 = y1.*(1-y1).*e1;
    
    % Adjust the weight of the neural network by hierarchy
    dW1 = alpha * delta1 * x';
    W1 = W1 + dW1;
    
    dW2 = alpha * delta * y1';
    W2 = W2 + dW2;
end

    