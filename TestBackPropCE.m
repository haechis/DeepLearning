clearvars all; close all;
%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% Practice: Hae-Chang Lee, gockdd1677@gmail.com
% 2021.09.01 
%% Chapter 3. Multi Layer Neural network
%% Example (3) Cross Entropy

%% Learning Data
% Input
% X -> c(n): n-th input data
X = [0, 0, 1;
    0, 1, 1;
    1, 0, 1;
    1, 1, 1];

% Answer
% answer of X's row
D = [0; 1; 1; 0];

% Init Weight Mat.
W1 = 2 * rand(4,3) - 1;
W2 = 2 * rand(1,4) - 1;


iter = 0;

N = 4;


% Need to modify.
% (참값에 가까워지지 않음)
while 1
    [W1, W2] = BackpropCE(W1, W2 ,X, D);
    for i = 1:N
        x = X(i,:)';
        v1 = W1*x;
        y1 = Sigmoid(v1);
        
        v = W2*y1;
        y(i) = Sigmoid(v);
    end
    errs_ = y - D';
    
    check  =  errs_ < 1e-2;
    
    if check(1) && check(2) && check(3) && check(4)
        fprintf('Iteration: %d\n',iter)
        break;
    end
    
     
    iter = iter + 1;
    if iter > 100000
        fprintf('Neuralnet not converging\n')
        break;
    end 
end
