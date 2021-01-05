%% TRAIN
%x = MergedImg;
%t = MergedLabel_one_hot;
x= MergedImg_NOISE;
t = MergedLabel_one_hot_NOISE;


% Choose a Training Function
% 'trainlm' 'trainbr' 'trainscg' 
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = [150, 100];
net = patternnet(hiddenLayerSize, trainFcn);
net.trainParam.max_fail = 5;

% Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';
train_cnt = 17000*3;
net.divideParam.trainRatio = train_cnt;
net.divideParam.valRatio = 60000 - train_cnt;
net.divideParam.testRatio = 3000;

% Choose a Performance Function
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

%% Test Acc + Plot

% Acc
tind = vec2ind(testLabelSmall_one_hot);
yind = vec2ind(net(testImgSmall));
test_acc = sum(tind == yind)/numel(tind)

% Plot
C = {'T-shirt/top','Trousers','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag', 'Ankle boot'};
I = [0,1,2,3,4,5,6,7,8,9];
plotconfusion(categorical(tind,I,C), categorical(yind,I,C), 'npr-tool 1xhidden (10 neurons)');

%% Save
save 'npr-tool-150-100.mat' net

%% PLOT 10 MISSCLASSIFIED IMAGES
miss = tind ~= yind;

good_labels = tind(miss);
bad_labels = yind(miss);
bad_images = testImgSmall(:,miss);

montage_cell = cell(1,10);
for idx = 1:1:10
    ii = idx+125;
    IM = imresize( reshape(bad_images(:,ii),28,28)  , 7)/255;
    box_color = {'green', 'red'};
    pos = [0 0;0 40]; 
    text = { C{ good_labels(ii) }, C{ bad_labels(ii) }};
    J = insertText(IM,pos,text,'FontSize',15, 'BoxOpacity',0.9, 'BoxColor', box_color);
    montage_cell{idx} = J;
end

montage(montage_cell)

