%% New weights are:
w_i_hb_new = [1.4079    0.3921   -0.0079
              -1.9913    0.7913   -0.6087];
   
w_h_ob_new = [2.0970   -1.0247    0.3749
              0.9758    0.8998   -0.5026];

%% Data and weights:

% Data
p = [-1;1];
d = [0.3 ; 0.3];

% Network weights
w_i_hb = [ 1.4 0.4 0.0 ; -2.0 0.8 -0.6 ];
w_h_ob = [ 2.1 -1.0 0.4 ; 1.0 1.1 -0.3 ];
lambda = 2;
%% Forward and Backward:

% Forward
y_h = logsig(lambda * w_i_hb * [p;1]);
y_o = logsig(lambda * w_h_ob * [y_h;1])
e1 = (y_o - d)'*(y_o - d)

% Backward
a = 1.5;

% 1. hidden  <--- output 
delta_o = (d-y_o) .* lambda .* y_o .* (1-y_o) % shape (2, 1)
grad1 = delta_o * [y_h;1]'  % shape (2, 3)
w_h_ob1 = w_h_ob + a * grad1 % shape (2, 3)

% 2. input  <--- hidden 
% s kazdou hodnout z delta_o vynasobit radek z w_h_ob a secist po radcich
A = sum(diag( delta_o') * w_h_ob, 2); % shape (2, 1)
delta_h =   A .* lambda .* y_h .* (1-y_h)  % shape (2, 1)
grad2 = delta_h * [p1; 1]'    % shape (2, 3)
w_i_hb1 = w_i_hb +  a * grad2 % shape (2, 3)

% Forward po druhe (err je mensi)
y_h = logsig(lambda * w_i_hb1 * [p;1])
y_o = logsig(lambda * w_h_ob1 * [y_h;1])
e1
e2 = (y_o - d)'*(y_o - d)
