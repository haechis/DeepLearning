function out = DeltaBatch(W,X,D)

alpha = 0.9; % Learning rate
dWsum = zeros(3,1);

N = size(X,1);

for k = 1:N
    x = X(k,:)';
    d = D(k);
    
    v = W*x;
    y = Sigmoid(v);
    
    e = d - y;
    delta = y*(1-y)*e;
    
    dW = alpha * delta * x;
    
    dWsum = dWsum + dW;
end
dWavg = dWsum / N;

out(1) = W(1) + dWavg(1);
out(2) = W(2) + dWavg(2);
out(3) = W(3) + dWavg(3);

end % end function.