%% LOAD DATASET
% trainImgSmall + trainLabelSmall ( 20,000 )
% testImgSmall + testLabelSmall   ( 3,000  )
% X: 784 values in [0, 255] (28x28 grayscale img)
load FMNISTSmall

% Permutate the training data...
perm = randperm(size(trainImgSmall,2));
trainImgSmall = trainImgSmall(:,perm);
trainLabelSmall = trainLabelSmall(perm);

% Merge it so that nf-tool can split it...
MergedImg =   [trainImgSmall testImgSmall];
MergedLabel = [trainLabelSmall testLabelSmall];

% Onehot encode...
testLabelSmall_one_hot = onehotencode(categorical(testLabelSmall), 1);
MergedLabel_one_hot = onehotencode(categorical(MergedLabel), 1);


%% Add noise to data

trainImgSmall1 = trainImgSmall + 3*randn(size(trainImgSmall));
trainImgSmall2 = trainImgSmall + 3*randn(size(trainImgSmall));
trainImgNoise = [trainImgSmall(:,1:17000) trainImgSmall1(:,1:17000) trainImgSmall2(:,1:17000)  trainImgSmall(:,17001:20000)  trainImgSmall1(:,17001:20000) trainImgSmall2(:,17001:20000)];
trainLabelNoise = [trainLabelSmall(1:17000) trainLabelSmall(1:17000) trainLabelSmall(1:17000)  trainLabelSmall(:,17001:20000) trainLabelSmall(:,17001:20000) trainLabelSmall(:,17001:20000)];
 
% 60k train 3k test
MergedImg_NOISE =   [trainImgNoise testImgSmall]; 
MergedLabel_NOISE = [trainLabelNoise testLabelSmall];
MergedLabel_one_hot_NOISE = onehotencode(categorical(MergedLabel_NOISE), 1);





%% TASK 1: nftool
% 1. plot test results using 'plotconfusion'

% 2. try: 15, 20, 30, 50, 100 neurons (results in table + discussion)

% 3. save trained network [save 'nf-tool-20.mat' net]

nftool

%% TASK 2: nprtool
% 1. achieve > 0.86 acc on test data (one hot encoding !!!)
%    label_onehot = full(inv2cet(T')')
%    output = argmax(output layer)

% 2. generate and modify script - 2 hidden layers (<200 neurons)
%    tip: layers=[100, 50] ; train on all 20K ; 100 iterations ; acc 0.87

% 3. test + 'plotconfusion'  +  [save 'task2.mat' net_2]

% 4. plot 10 wrongly classified with t and y (why you thing it is wrong?)
%    colormap(gray)
%    image(reshape(trainImgSmall(:,2217),28,28))

nprtool







