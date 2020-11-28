function LPar = Memorizer(Tr, DTr, ~)
% The learning algorithm which only remembers all training
% samples and the desired answers for them

% inputs:
% Tr matrix with training samples in columns
% DTr row vector of the desired outputs
% Par is not used here, it is present here for
% compatibility only
%
% output:
% LPar a cell array containing the training samples and
% the desired outputs for them
LPar = {Tr,DTr};

end

