%% Task 2:

% Input --> Hidden Layer
h1 = [5 5 ; -8 -8];
b1 = [-7 ; 3];
  
% Hidden --> Output Layer
h2 = [-7 3 ; 8  8];
b2 = [5 ; -5];

% Results
p1 = [0;0]; % 1;1
p2 = [0;1]; % 1;0
p3 = [1;0]; % 1;0
p4 = [1;1]; % 0;1

o1 = logsig([h2 b2] * [logsig([h1 b1] * [p1;1]);1])
o2 = logsig([h2 b2] * [logsig([h1 b1] * [p2;1]);1])
o3 = logsig([h2 b2] * [logsig([h1 b1] * [p3;1]);1])
o4 = logsig([h2 b2] * [logsig([h1 b1] * [p4;1]);1])