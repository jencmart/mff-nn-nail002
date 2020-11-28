function c = perc_recall(p,x)
% A function simulating Perceptron
    newX = [x;ones(1,size(x,2))];
    potential = p * [newX];
    c = potential > 0;
end
