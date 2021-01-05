%% Assignment 3
%% Task 1
%%
% $$x_i \sim N(0,\sigma^2) $$
%%
% $$ E[x_i] = 0 $$
%%
% $$ \sigma^2_{x_i} = \sigma^2 $$

%%
% ---------------------
%%
% $$ w_i \sim Uniform(-a, a) $$
%%
% $$ E[w_i] = 0$$
%%
% $$ \sigma^2_{w_i} = (a-(-a))^2/12 = (2a)^2/12 = a^2/3 $$
%%
% ---------------------
%%
% $$ \xi = \sum_i^n w_i  x_i $$
%% Goal
%%
% $$ E[\xi] = 0 $$
%%
% $$ \sigma^2_\xi = 1 $$
%% Solution for zero mean
% $$ E[\xi] = \sum E[x_i w_i] = \sum E[x_i]E[w_i] = 0$$
%%
% First equality from lienarity of E
%%
% Second equality from independence of $$x_i$ and $$w_i$
%% 
% So $$E[\xi]= 0 $$ and does not depend on $$a$
%% Solution for unary variance
% $$ \sigma^2_\xi = E[\xi^2] - E^2[\xi] = E[\xi^2] = E[ (\sum_i^n w_i
% x_i) (\sum_i^n w_i x_i) ] =  E[ \sum_{i,j}^n w_i w_j x_i x_j ] = 
% \sum_{i,j}^n E[ w_i w_j x_i x_j ] =  \sum_{i,j}^n E[ w_i w_j] E[x_i x_j] =
% \sum_{i}^n E[w_i^2] E[x_i^2] $$
%%
% $$E[x_i^2] = \sigma^2 $$ 
%%
% $$E[w_i^2] = a^2/3 $$ 
%%
% $$ \sigma^2_\xi = \sum_{i}^n E[w_i^2] E[x_i^2] = n \sigma^2 a^2/3 $$
%%
% We want $$ \sigma^2_\xi = 1 $$ thus
%%
% $$ n \sigma^2 a^2/3 = 1 $$
%%
% $$ a = (3/(n\sigma))^{1/2} = (3/n)^{1/2} 1/\sigma  $$

