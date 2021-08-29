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
D = [0; 0; 1; 1];

% Init Weight Mat.
W1 = 2 * rand(4,3) - 1;
W2 = 2 * rand(1,4) - 1;

for epoch = 1:10000 % train
[W1, W2] = BackpropXOR(W1, W2, X, D);
end

