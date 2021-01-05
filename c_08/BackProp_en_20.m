%% Multi-Layered Neural Networks and the Backpropagation Algorithm
%
%% Remark
% Working with neural networks, we use the representation of vectors and
% weights the same as in the MATLAB, in particular within the Deep Learning
% Toolbox:
%
% _*Input vectors are always column vectors.*_
% 
%
%
% For easy computing potential on a neuron, the weights of incoming
% synapses of the neuron are stored as a row vector.
% 
%% Multi-Layered Neural Network - Recall
% Let us take a neural network with the topology [2,2,1], i.e., the network
% has 2 input neurons, 2 hidden neurons in a single hidden layer, and one
% output neuron. Let the weights of synapses between the input and the
% hidden layer be in the following matrix:
w_i_h = [ 0.50000 1.50000
         -0.50000 0.50000];
%% 
% |w_i_h(i,j)| is the weight of the synapse from the input |j| into the
% hidden neuron |i|. I.e., each row of the weight matrix corresponds to
% weights of synapses leading *to* the neuron!
%%
% Let the synaptic weights between the hidden and the output layer
% be in the matrix:
w_h_o = [2.00000 -1.00000];
%%
% |w_h_o(1,i)| is the weight of the connection from the hidden neuron |i| 
% to the output neuron. thresholds of the hidden neurons are in the matrix: 
b_h = [ 0.00000
        0.50000];
%% 
% and the threshold of the outout neuron is:
b_o = -0.50000;
%%
% Hence the weights from the input layer into the hidden layer with added 
% virtual neuron with fixed output 1 (for representing thresholds) are:
w_i_hb=[w_i_h b_h]
%% 
% The weights from the hidden layer into the output layer
% with added virtual neuron with output 1 are:
w_h_ob=[w_h_o b_o]
%%
% A sigmoidal transfer function is implemented in MATLAB (in its Neural
% Networks Toolbox)
%
% $$logsig(x) = \frac{1}{1 + e^{-x}}.$$
% 
% It is the sigmoid function with the slope $\lambda = 1$.
%% Tasks:
% 1. Compute output of net for inputs 
p1 = [-1; 1];

y_h = logsig( w_i_hb * [p1; 1] )
y_o = logsig( w_h_ob * [ y_h ; 1 ] )



% 2. p1=[-1; 1]| y1 = 0.9 What is the error of the net on this pattern?

d = 0.9;
diff = y_o - d;
err1 = 1/2 * transpose(diff) * diff  % jeste by to mela byt suma pres vsecny patterny...
%  0.0905


% 3. What will be weights after 1 step of backprop. with lr=0.2?
% no momentum - so no alpha_m part ...
lambda = 1
a = 0.2



% >>>>> hiden -> output <<<<<<
delta_o = (d-y_o) .* lambda * y_o .* (1-y_o) % ale tady mam 1 hodnotu... ( co kdyz budou 2?)
% chci mit 3 hodnoty  1x3...
%  d * y_h
   % * % % %  = % % %
grad1 = delta_o * [y_h;1]' % chci mit 3 hodnoty...
w_h_ob1 = w_h_ob + a * grad1 % 2.0132 -0.9852 -0.4819

w_h_ob % ... 3 hodnoty v radku ...
% >>>>> input -> hidden <<<<<<
% todo -- s kazdou hodnout z delta_o vynasobit radek z w_h_ob
% asi takhle: 
%    A = diag( delta_o' ) * w_h_ob 
%    sum(A,2) % ... sum over rows ...
delta_h =   sum(delta_o * w_h_ob) .* lambda * y_h .* (1- y_h)
% delta_h = delta_o * w_h_ob
% ...chceme 6 hodnot 2x3  ...
% d * p
  % * % % %  = % % %
  %            % % % 
grad2 = delta_h * [p1; 1]'
w_i_hb1 = w_i_hb +  a * grad2

% new error is smaller !
y_h = logsig( w_i_hb1 * [p1; 1] )
y_o = logsig( w_h_ob1 * [ y_h ; 1 ] )
diff = y_h - y_o
err2 = 1/2 * transpose(diff) * diff  % jeste by to mela byt suma pres vsecny patterny...
% 0.0542
