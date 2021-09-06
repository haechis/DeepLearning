clearvars all;
%% 딥러닝 첫걸음
%% 한빛미디어, 김성필
%% Practice: Hae-Chang Lee, gockdd1677@gmail.com
% 2021.09.06 
%% Chapter 4. Multi Layer Neural network
%% Example (2) MultiClass
% 결과: (1) 1번 데이터: 1,4 헷갈림
%       (2) 3번 데이터: 2,3 중에 다소 모호한데, 한 픽셀차이로 3으로 분류
%       (3) 4번 데이터: 인간의 관점에서는 5이지만, 3, 5 분류가 모호함       


TestMultiClass; % 학습을 시킴

X(:,:,1) = [ 0 0 1 1 0;
             0 0 1 1 0;
             0 1 0 1 0;
             0 0 0 1 0;
             0 1 1 1 0;];
         
X(:,:,2) = [ 1 1 1 1 0;
             0 0 0 0 1;
             0 1 1 1 0;
             1 0 0 0 1;
             1 1 1 1 1;];
         
X(:,:,3) = [1 1 1 1 0;
            0 0 0 0 1;
            0 1 1 1 0;
            1 0 0 0 1;
            1 1 1 1 0;];
        
X(:,:,4) = [ 0 1 1 1 0;
             0 1 0 0 0;
             0 1 1 1 0;
             0 0 0 1 0;
             0 1 1 1 0;];
         
X(:,:,5) = [ 0 1 1 1 1;
             0 1 0 0 0;
             0 1 1 1 0;
             0 0 0 1 0;
             1 1 1 1 0;];
         
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

