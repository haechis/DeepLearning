clearvars all; close all;
%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% Practice: Hae-Chang Lee, gockdd1677@gmail.com
% 2021.08.29 
%% Chapter 3. Multi Layer Neural network
%% Example (1) XOR Problem

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

for epoch = 1:10000 % train
[W1, W2] = BackpropXOR(W1, W2, X, D);
end

N = 4;
y = zeros(N,1);
for k = 1:N
    x = X(k,:)';
    v1 = W1*x;
    y1 = Sigmoid(v1);
    v = W2 * y1;
    y(k) = Sigmoid(v);
end

fprintf(1,'<Multi-Layer Neural Network>\n<True> %d, %d, %d, %d \n',D(1),D(2),D(3),D(4))

fprintf(1,'<Estimated> %6.4f, %6.4f, %6.4f, %6.4f',y(1),y(2),y(3),y(4))
