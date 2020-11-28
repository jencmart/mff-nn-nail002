function LPar = PLearn(X,y,Par)
p = perc_learn(Par{1},X,y,Par{2},Par{3});
LPar = {p};
end