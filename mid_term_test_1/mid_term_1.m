%% Data and weights:

% Data
p = [1 ; 1 ; 1];
d = [0.9 ; 0.3];

% Network weights
%w_i_hb = [1.4 0.8 -2.0 0.0];
      
%w_h_ob = [ 1.0 0.2     % out_1
%          -1  -0.2 ];  % out_2
%% Forward and Backward:

% Forward
y_h = logsig( w_i_hb * [p;1]);
y_o = logsig( w_h_ob * [y_h;1])
e1 = (y_o - d)'*(y_o - d)

% Backward
a = 0.4;
% 1. hidden  <--- output 
delta_o = (d-y_o)  .* y_o .* (1-y_o) % shape (2, 1)
grad1 = delta_o * [y_h;1]'  % shape (2, 2) -- protoze H->OUT je 2x2
w_h_ob1 = w_h_ob + a * grad1 % shape (2, 2) -- protoze H->OUT je 2x2

% 2. input  <--- hidden 
% s kazdou hodnout z delta_o vynasobit radek z w_h_ob ( a secist po radcich [2])
% d_[0] = sum_k (d_k * w_jk) .. delta_o[0] * w_h_ob[0,0] + delta_o[] * w_h_ob[0,1]
A = sum(diag(delta_o) * w_h_ob, 'all') % shape (, 1) -- protoze IN-> H je 1x1
% delta_o ?? w_h_ob
delta_h =   A .* y_h .* (1-y_h) % A * cislo * cislo
grad2 = delta_h * [p; 1]'    % shape (1, 1) * (1, 4) = (1, 4)
w_i_hb1 = w_i_hb +  a * grad2 % shape (1, 4)

% Forward po druhe (err je mensi)
y_h = logsig( w_i_hb1 * [p;1])
y_o = logsig( w_h_ob1 * [y_h;1])
e1
e2 = (y_o - d)'*(y_o - d)
w_i_hb = w_i_hb1
w_h_ob = w_h_ob1
% 0492
