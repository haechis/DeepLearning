clearvars all; close all;
%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% Practice: Hae-Chang Lee, gockdd1677@gmail.com
% 2021.09.06 
%% Chapter 4. Multi Layer Neural network
%% Example (1) MultiClass

rng(3);

x = zeros(5,5,5);

X(:,:,1) = [0 1 1 0 0; % 1
            0 0 1 0 0;
            0 0 1 0 0;
            0 0 1 0 0;
            0 1 1 1 0;];
        
X(:,:,2) = [ 1 1 1 1 0;
             0 0 0 0 1;
             0 1 1 1 0;
             1 0 0 0 0;
             1 1 1 1 1;];
         
X(:,:,3) = [ 1 1 1 1 0;
             0 0 0 0 1;
             0 1 1 1 0;
             0 0 0 0 1;
             1 1 1 1 0;];
         
X(:,:,4) = [ 0 0 0 1 0; 
             0 0 1 1 0;
             0 1 0 1 0;
             1 1 1 1 1;
             0 0 0 1 0;];
         
X(:,:,5) = [ 1 1 1 1 1;
             1 0 0 0 0;
             1 1 1 1 0;
             0 0 0 0 1;
             1 1 1 1 0;];
         
D = [ 1 0 0 0 0;
      0 1 0 0 0;
      0 0 1 0 0;
      0 0 0 1 0;
      0 0 0 0 1;];
  
W1 = 2*rand(50,25) - 1;
W2 = 2*rand(5, 50) - 1;

for epoch = 1:10000 % train
    [W1, W2] = MultiClass(W1, W2, X, D);
end

N = 5; % inference
ans = zeros(5,5);
for k = 1:N
    x = reshape(X(:,:,k),25,1);
    v1 = W1 * x;
    y1 = Sigmoid(v1);
    v = W2 * y1;
    y = Softmax(v);
    ans(:,k) = y;
end

% answer 확인은 one-hot 인코딩된 ans  확인.