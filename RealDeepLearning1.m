clearvars all;
%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% Practice: Hae-Chang Lee, gockdd1677@gmail.com
% 2021.09.06
%% Chapter 5. DeepLearning
%% Example (#) Practice

TestDeepReLU;

X(:,:,1) = [ 0 0 1 1 0;
    1 0 1 1 0;
    0 1 0 1 0;
    0 0 0 1 0;
    0 1 1 1 0;];

X(:,:,2) = [ 1 1 1 1 0;
    0 0 0 0 1;
    0 1 1 1 0;
    1 0 1 0 1;
    1 1 1 1 1;];

X(:,:,3) = [1 1 1 1 0;
    0 0 1 0 1;
    0 1 1 1 0;
    1 0 0 0 1;
    1 1 1 1 0;];

X(:,:,4) = [ 0 1 1 1 0;
    0 1 1 0 0;
    0 1 1 1 0;
    0 0 0 1 0;
    0 1 1 1 0;];

X(:,:,5) = [ 0 1 1 1 1;
    0 1 0 0 1;
    0 1 1 1 0;
    0 0 0 1 0;
    1 1 1 1 0;];

N = 5; % inference
ans = zeros(5,5);
for k = 1:N
    x = reshape(X(:,:,k),25,1);
    v1 = W1 * x;
    y1 = ReLU(v1);
    
    v2 = W2*y1;
    y2 = ReLU(v2);
    
    v3 = W3*y2;
    y3 = ReLU(v3);
    
    v = W4*y3;
    y = Softmax(v);
    ans(:,k) = y; % 완벽하게는 찾아내지 못하는 모습. 물론 모호하긴 함.
end


