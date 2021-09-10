%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% Practice: Hae-Chang Lee, gockdd1677@gmail.com
 
%% Chapter 2. Neural network
%% Example (2) SGD (Stochastic Gradient Descent)
% batch 
%
%% Learning Data
% Input
% X -> c(n): n-th input data
X = [0, 0, 1;
    0, 1, 1;
    1, 0, 1;
    1, 1, 1];

% Answer
% answer of X's row
D = [0; 0; 1; 1];

% Weight Initialization
% simply set to 1
W = [1 1 1];


iter = 0;

while 1
    W = DeltaBatch(W,X,D);
    for i = 1:N
        x = X(i,:)';
        v = W*x;
        res(i) = Sigmoid(v);
    end
    errs_ = res - D';
    
    check  =  errs_ < 1e-2;
    
    if check(1) && check(2) && check(3) && check(4)
        break;
    end
    
     
    iter = iter + 1;
    if iter > 100000
        fprintf('Neuralnet not converging\n')
        break;
    end 
end

fprintf(1,'<Inferenced Result> \nD(1): %6.4f, D(2): %6.4f, D(3): %6.4f, D(4): %6.4f\n', res(1),res(2), res(3), res(4))

fprintf(1,'<Errors (%s)> \n D(1): %6.4f, D(2): %6.4f, D(3): %6.4f, D(4): %6.4f\n\n', ...
    '%',abs(res(1)-D(1))*100,abs(res(2)-D(2))*100,abs(res(3)-D(3))*100,abs(res(4)-D(4))*100)

fprintf('<Converging Speed> \nIteration: %d\n',iter)
