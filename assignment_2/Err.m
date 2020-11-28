function E = Err(Train, Predict, Params, Train_X, Train_y, Test_X, Test_y)

% Params   = cell array with train algo. parameters
% Train_X  = training set (training vectors are the columns of Tr),
% Train_y  = row vector of desired outputs, and

% Train the params
%LPar = perc_learn(Par,Train_X,Train_y,0.5,100) ;
LParams = feval(Train, Train_X, Train_y, Params);

% Predict the results
% y_hat = perc_recall(LParams, Test_X);
y_hat = feval(Predict, LParams, Test_X);

% calculate the error 〈0; 1〉 
E = sum(abs(y_hat-Test_y)) / length(Test_y);
end

