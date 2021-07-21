clearvars all; close all;
%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% Practice: Hae-Chang Lee, gockdd1677@gmail.com
 
%% Chapter 2. Neural network
%% Example (3) SGD (Stochastic Gradient Descent)
%  SGD vs batch
%

% Input data
X = [0 0 1;
    0 1 1;
    1 0 1;
    1 1 1;];

% true values
D = [0;0;1;1];

n = 1000;

% sum of squared errors -> Mean
E1 = zeros(n,1);
E2 = zeros(n,2);

% Initialzation Weighting
W1 = 2*rand(1,3) - 1;
W2 = W1;

N = 4;

for epoch = 1:n % train!
    W1 = DeltaSGD(W1,X,D);
    W2 = DeltaBatch(W2,X,D);
    
    % es1, es2 : sum of squared errors
    es1 = 0;
    es2 = 0;
    N = 4;
    
    for k = 1:N
        x = X(k,:)';
        d = D(k);
        
        v1 = W1*x;
        y1 = Sigmoid(v1);
        es1 = es1 + (d - y1)^2;
        
        v2 = W2*x;
        y2 = Sigmoid(v2);
        es2 = es2 + (d - y2)^2;
    end
    E1(epoch) = es1/N;
    E2(epoch) = es2/N;
end

figure;
hold on;
plot(E1,'r')
plot(E2,'b:')
xlabel('epoch')
ylabel('Average of Training error')
legend('SGD','Batch')
        
    