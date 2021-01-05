%% TRAIN
x = MergedImg;
t = MergedLabel;

% Choose a Training Function
trainFcn = 'trainscg'; 

% Create a Fitting Network
hiddenLayerSize = 10; % try: [10] [15], [20] , [30], [50], [100]
net = fitnet(hiddenLayerSize,trainFcn);
net.trainParam.max_fail = 10;

% Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';
net.divideParam.trainRatio = 17000;
net.divideParam.valRatio = 20000 - net.divideParam.trainRatio;
net.divideParam.testRatio = 3000;

% Choose a Performance Function
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
% [net,tr] = train(net,x,t,'useGPU','yes');
% 'gpuDevice' command to check GPU
[net,tr] = train(net,x,t);

%% Test Acc + Plot

% acc    trainscg  trainlm
% 10 ... 0.5740    0.5343
% 15 ... 0.5050    0.4877
% 20 ... 0.5240    0.3460 
% 30 ... 0.4913    0.3500
% 50 ... 0.5523
y_test = max(0, min(9, round(net(testImgSmall))));
test_acc = nnz(~( (testLabelSmall+1)- (y_test+1))) / length(testLabelSmall)


% Plot
C = {'T-shirt/top','Trousers','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag', 'Ankle boot'};
I = [0,1,2,3,4,5,6,7,8,9];
plotconfusion(categorical(testLabelSmall,I,C), categorical(y_test,I,C), 'nf-tool 1xhidden (10 neurons)');


%% Save
save 'nf-tool-10.mat' net
