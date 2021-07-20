%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% 실습: 이해창, gockdd1677@gmail.com
%
%

%% Chapter 2. 신경망
%% Example (1) SGD (Stochastic Gradient Descent)
%
%% Learning Data
% Input
X = [0, 0, 1;
    0, 1, 1;
    1, 0, 1;
    1, 1, 1];
% Answer
D = [0; 0; 1; 1];

% Weight Initialization
W = [1 1 1];
% Weight Update
for epoch = 1:10000
    W = DeltaSGD(W,X,D);
end

results = zeros(1,4);
for k = 1:4
   x = X(k,:)';
   v = W*x;
   results(k) = Sigmoid(v);
end

fprintf(1,'<Inferenced Result> \nD(1): %6.4f, D(2): %6.4f, D(3): %6.4f, D(4): %6.4f\n', results(1),results(2), results(3), results(4))