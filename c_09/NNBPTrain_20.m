%% Training Multi-Layered Neural Networks in MATLAB
% See
% <http://www.mathworks.com/help/nnet/ug/multilayer-neural-networks-and-bac
% kpropagation-training.html?searchHighlight=backpropagation Multilayer
% Neural Networks and Backpropagation Training>.
%
% Multi-layer neural networks are designed and applied in 7 steps
%
% # 1. Collect data
% # 2. Create the network
% # 3. Configure the network
% # 4. Initialize the weights and biases
% # 5. Train the network
% # 6. Validate the network (post-training analysis)
% # 7. Use the network (and data transformations...)
%
%% 1. Collecting data
% Here we can use various preprocessing functions (e.g. normalization).
% Input and target (desired output) vectors are column vectors.
p = [-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10
      -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 1 1 1 1 1 1 1 1 1  1];
t = [10   9  8  7  6  5  4  3  2  1 0 1 2 3 4 5 6 7 8 9 10];
%% 2. Creating network
% without params: hiddenSizes=[10] and trainFunc='trainlm'

% with params   : feedforwardnet(hiddenSizes, trainFunc)   
%    trainFunc:
%       'gdm' ... gradient descent with momentum
%       'trainlm' ... Levenberg-Marquardt Method (2.derivace ...)
net = feedforwardnet;

%% 3. Configuring network 
% Next, we configure the network by providing the X ... |p|  and Y ... |t|
% #inputs and #output neurons created automatically from the shapes of X, Y
% also it initializes the weights and biases of the network
net = configure(net, p, t);

%% 4. Re-initializing weights and biases
% If we want to re-initialize the network. For this we can call function |init|
net = init(net);

%% 5a. Training
% * |train| batch training - update weights/biases by average from batches
% * |adapt| incremental learning - update weights/biases after each pattern
%  * batch training is more efficient (is just better...)

% When using train(), it is not necessary to use configure()
% train() calls configure() if the network is not initialized

% don't show train window: set net.trainParam.showWindow = false
% show it in command line: set net.trainParam.showCommandLine = true

% During training we can watch:
%  error function (button |Performance|)
%  training state (button |Training State|)

% net = adapted network in the first returned value
% tr  = log of the training
[net,tr]=train(net, p, t);
%% 5b. When to stop training:
% * |net.trainParam.min_grad| minimum gradient magnitude - if reached, stop
%
% * |net.trainParam.max_fail| the number of successive iterations that the 
%   validation performance fails to decrease - if reached, the training
%   stops and the returned network is that obtained before the validation
%   performance started to increase (this should prevent overfitting),
%
% * |net.trainParam.time| is the maximum allowed training time
%
% * |net.trainParam.goal| is the target performance value; the default 
%   performance function of a network is |mse| - mean square error; the
%   default value for the |goal| is 1.0e-5.
%
% * |net.trainParam.epoch| is the maximum number of sweeps through the
%   whole training set.

% For instance, we can set
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-6;

% ... and continue training of the network
[net,tr]=train(net,p,t);

%% 5c. Training window:
% This window shows that the data has been divided using the |dividerand|
% function, and the Levenberg-Marquardt (|trainlm|) training method has been
% used with the mean square error performance function (|mse|). Recall that these
% are the default settings for |feedforwardnet|.

% During training, the progress is constantly updated in the training
% window. Of most interest are the performance, the magnitude of the
% gradient of performance and the number of validation checks. The
% magnitude of the gradient and the number of validation checks are used to
% terminate the training. The gradient will become very small as the
% training reaches a minimum of the performance. If the magnitude of the
% gradient is less than 1e-5, the training will stop. This limit can be
% adjusted by setting the parameter |net.trainParam.min_grad|. The number of
% validation checks represents the number of successive iterations that the
% validation performance fails to decrease. If this number reaches 6 (the
% default value), the training will stop. In this run, you can see that the
% training did stop because of the number of validation checks. You can
% change this criterion by setting the parameter |net.trainParam.max_fail|.
% (Note that your results may be different than those shown in the
% following figure, because of the random setting of the initial weights
% and biases.)

%% 6a. Post training analysis of the network
tr

% We can see indices of input patterns which were used 
% training set   - |tr.trainInd|
% validation set - |tr.valInd| 
% testing set    - |tr.testInd|

% Additional training of net with the same train/val/test data split set:
%   net.divideFcn = 'divideInd'
%   net.divideParam.trainInd = tr.trainInd
%   net.divideParam.valInd = tr.valInd
%   net.divideParam.testInd = tr.testInd

% tr contains log of several variables during the course of training
% such as:
%  value of the performance function
%  the magnitude of the gradient
%  etc...

% Plot training record:
plotperf(tr)

% We can also train without storing information on training...
%  net = train(net,p,t);  
  
%% 6b. Regression Plot

% The next step in validating the network is to create a regression plot,
% which shows the relationship between the outputs of the network and the
% targets. 
% After training we can plot regression graphs (the button |Regression|).

%     \includegraphics[width=10.5cm]{regres}

% If the training were perfect, the network outputs and the
% targets would be exactly equal, but the relationship is rarely perfect in
% practice.

%% 7a. Applying the trained network
sim(net, [2;3])

%% 7b. Data Transformations
%  To improve perfomance of NN, we can trasform the data
% Various normalization functions are implemented within the NN Toolbox
%
% Some networks do preprocessing automatically (feed-forward NN)
%    replace missing values |NaN| in the input with avg for the respective attributes
%    leave out constant rows
%    normalize the input values using min-max normalization to [-1,1]
%
%
% This is stored in the network object 
net.inputs{1}.processFcns
%   index 1 refers to the first input vector. 
%   ... as there is only one input vector for the feedforward network
%   |i|-th input processing function can have parameters net.inputs{1}.processParams{i}
%
%
% Similarly, outputs of the network are processed with the functions
net.outputs{2}.processFcns
%   where the index 2 refers to the output vector coming from the second layer. 
%   we have only one output vector, and it comes from the final layer...

% Note that the same transformations must be done also on new data...

%% 7c. Min-max normalization to [-1,1]  (toolbox to dela automaticky..)

% it can be applied on input and also for output data
[pn,ps] = mapminmax(p);
[tn,ts] = mapminmax(t);
net = train(net,pn,tn);  % trainig on normalized data ...

% |pn| and |tn| are the transformed input and target patterns, respective. 
% The returned |ps| and |ts| are the parameters of the respective 
% transformations, which will be used for transformations on new data and 
% for the inverse transformation of outputs. 

%% 7d. after prediction on new data, we mut transform output back...
an = sim(net,pn);
a = mapminmax('reverse', an, ts);

% On new data, we also use the same transformation...
pnew = [ 1 2
         0 1];
     
pnewn = mapminmax('apply',pnew,ps);
anewn = sim(net,pnewn);
anew = mapminmax('reverse',anewn,ts);

%% Jsou i jine normalizace (Normalization by standard deviation)
% Normalize to mu=0, var(and std)=1 
[pn,ps] = mapstd(p);
[tn,ts] = mapstd(t);


an = sim(net,pn);
a = mapstd('reverse',an,ts);

pnewn = mapstd('apply',pnew,ps);
anewn = sim(net,pnewn);
anew = mapstd('reverse',anewn,ts);

%% Jsou i jine normalizace (Principal Component Analysis)

% At first we must normalize to mu=0, var(and std)=1  
[pn,ps1] = mapstd(p);

% Then we can perform PCA
[ptrans,ps2] = processpca(pn,0.02); 
% odeber dimenze ktere prispivaji mene nez 0.02% do variance 
% [muze a nemusi neco odebrat..]


% We must perform the same transformation on a new data
pnewn = mapstd('apply',pnew,ps1);
pnewtrans = processpca('apply',pnewn,ps2);
a = sim(net,pnewtrans);

%% >> PRIKLAD <<

% data
x = -3:0.2:3  % [-3. , -2.8, ..., 3]
y = 3*sin(2*x-1)+3

% network
net = feedforwardnet([10,10]);

% training
net1 = train(net, x, y);

% nice plot
xx = -5:0.1:5;
plot(x,y,'r',xx, sim(net1,xx), 'b+')
